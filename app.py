from flask import Flask, request, render_template
import pandas as pd
import os
import msgpack
import numpy as np
from datetime import datetime
import math

app = Flask(__name__)

CACHE_FILE = "ph_wca_rankings.msgpack"
EVENTS = {
    '333': '3x3', '222': '2x2', '444': '4x4', '555': '5x5', '666': '6x6', '777': '7x7',
    '333bf': '3BLD', '333fm': 'FMC', '333oh': 'OH', 'clock': 'Clock', 'minx': 'Mega',
    'pyram': 'Pyra', 'skewb': 'Skewb', 'sq1': 'Sq1', '444bf': '4BLD', '555bf': '5BLD',
    '333mbf': 'MBLD'
}

def format_wca_time(value, event_id):
    """Formats WCA integers into readable strings (e.g., 71742 -> 7:17.42)"""
    if value <= 0: return ""
    if event_id == '333fm': return str(value)
    if event_id == '333mbf':
        s = str(value).zfill(10)
        solved = 99 - int(s[0:2])
        missed = int(s[8:10])
        total = solved + missed
        time_sec = int(s[2:7])
        return f"{solved}/{total} {time_sec // 60}:{str(time_sec % 60).zfill(2)}"
    
    centiseconds = value % 100
    seconds = (value // 100) % 60
    minutes = (value // 6000) % 60
    hours = (value // 360000)

    res = f"{seconds}.{str(centiseconds).zfill(2)}"
    if minutes > 0 or hours > 0:
        res = f"{minutes}:{str(seconds).zfill(2)}.{str(centiseconds).zfill(2)}"
    if hours > 0:
        res = f"{hours}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}.{str(centiseconds).zfill(2)}"
    return res

def decode_mbld_vectorized(series):
    valid_mask = series.notna() & (series > 0)
    scores = pd.Series(0.0, index=series.index)
    if not valid_mask.any(): return scores
    s = series[valid_mask].astype(np.int64).astype(str).str.zfill(10)
    solved = 99 - s.str[0:2].astype(int)
    missed = s.str[8:10].astype(int)
    time_sec = s.str[2:7].astype(int)
    scores[valid_mask] = (solved - missed) + ((3600 - time_sec) / 3600)
    return scores

def load_and_cache():
    # 1. Try loading from cache first
    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = msgpack.unpackb(f.read(), raw=False)
                if 'all_results' in data:
                    df_all = pd.DataFrame(data.get('all_results', []))
                    # Convert strings back to Timestamps for internal consistency
                    if not df_all.empty and 'date' in df_all.columns:
                        df_all['date'] = pd.to_datetime(df_all['date'])
                        
                    return (pd.DataFrame(data['names']), 
                            pd.DataFrame(data['single']), 
                            pd.DataFrame(data['average']), 
                            data.get('last_update', 'Unknown'), 
                            pd.DataFrame(data.get('podiums', [])), 
                            df_all)
        except Exception as e:
            print(f"Cache read error: {e}")

    # 2. If no cache or old cache, load from TSVs
    try:
        required_files = ["WCA_export_results.tsv", "WCA_export_ranks_single.tsv", 
                          "WCA_export_ranks_average.tsv", "WCA_export_competitions.tsv"]
        for f in required_files:
            if not os.path.exists(f):
                print(f"CRITICAL ERROR: {f} is missing!")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "File Error", pd.DataFrame(), pd.DataFrame()

        export_date = datetime.fromtimestamp(os.path.getmtime("WCA_export_results.tsv")).strftime('%b-%d, %Y')
        
        # Load Results
        res_df = pd.read_csv("WCA_export_results.tsv", sep="\t", 
                             usecols=['person_id', 'person_name', 'person_country_id', 'pos', 'round_type_id', 'event_id', 'best', 'average', 'competition_id'], 
                             low_memory=False)
        
        # Load Competitions
        comps = pd.read_csv("WCA_export_competitions.tsv", sep="\t", usecols=['id', 'year', 'month', 'day', 'name'])
        comps['date'] = pd.to_datetime(comps[['year', 'month', 'day']])
        
        # Filter for Philippines and merge with competition data
        ph_res_all = res_df[res_df['person_country_id'] == 'Philippines'].merge(comps[['id', 'date', 'name']], left_on='competition_id', right_on='id')
        ph_names = ph_res_all.drop_duplicates('person_id')[['person_id', 'person_name']]
        ph_ids = ph_names['person_id'].unique()

        # Load Ranks
        s_ranks = pd.read_csv("WCA_export_ranks_single.tsv", sep="\t", usecols=['person_id', 'event_id', 'best', 'country_rank'], low_memory=False)
        a_ranks = pd.read_csv("WCA_export_ranks_average.tsv", sep="\t", usecols=['person_id', 'event_id', 'best', 'country_rank'], low_memory=False)
        
        ph_s_ranks = s_ranks[s_ranks['person_id'].isin(ph_ids)]
        ph_a_ranks = a_ranks[a_ranks['person_id'].isin(ph_ids)]

        # CRITICAL: Convert date to string before msgpack serialization
        # This prevents the "can not serialize 'Timestamp' object" error
        ph_res_all_serializable = ph_res_all.copy()
        ph_res_all_serializable['date'] = ph_res_all_serializable['date'].dt.strftime('%Y-%m-%d')

        # Save to Cache
        cache_data = {
            'names': ph_names.to_dict('records'),
            'single': ph_s_ranks.to_dict('records'),
            'average': ph_a_ranks.to_dict('records'),
            'podiums': ph_res_all[(ph_res_all['pos'] <= 3) & (ph_res_all['round_type_id'].isin(['f', 'c']))][['person_id', 'event_id', 'pos']].to_dict('records'),
            'all_results': ph_res_all_serializable[['person_id', 'event_id', 'pos', 'round_type_id', 'date', 'name', 'best', 'average']].to_dict('records'),
            'last_update': export_date
        }
        
        with open(CACHE_FILE, 'wb') as f:
            f.write(msgpack.packb(cache_data))

        return ph_names, ph_s_ranks, ph_a_ranks, export_date, pd.DataFrame(cache_data['podiums']), ph_res_all
        
    except Exception as e:
        print(f"Error loading TSVs: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Error", pd.DataFrame(), pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_events = request.form.getlist('events')
        rank_type = request.form.get('rank_type', 'single')
    else:
        selected_events = request.args.getlist('events')
        rank_type = request.args.get('rank_type', 'single')

    if not selected_events: selected_events = ['333']
    page, per_page = int(request.args.get('page', 1)), 50
    leaderboard = []

    if rank_type == 'streak':
        if not ph_all_results.empty:
            event_id = selected_events[0]
            # Filter for Finals/Combined rounds only
            df_ev = ph_all_results[(ph_all_results['event_id'] == event_id) & 
                                   (ph_all_results['round_type_id'].isin(['f','c']))].sort_values(['person_id', 'date'])
            
            streaks = []
            for pid, group in df_ev.groupby('person_id'):
                max_s, curr_s = 0, 0
                # Temp variables for current running streak
                ts_comp, ts_val, te_comp, te_val = "", "", "", ""
                # Final variables for the longest recorded streak
                fs_comp, fs_val, fe_comp, fe_val = "", "", "", ""
                
                for _, row in group.iterrows():
                    # Determine the winning result (Average if exists, else Best)
                    res_value = row['average'] if row['average'] > 0 else row['best']
                    
                    # VALID WIN CHECK: Must be 1st place AND not a DNF/DNS (res_value > 0)
                    if row['pos'] == 1 and res_value > 0:
                        val_str = format_wca_time(res_value, event_id)
                        if curr_s == 0: 
                            ts_comp, ts_val = row['name'], val_str
                        curr_s += 1
                        te_comp, te_val = row['name'], val_str
                    else:
                        # Streak broken (either by a loss or a DNF win)
                        if curr_s >= max_s and curr_s > 0:
                            max_s, fs_comp, fs_val, fe_comp, fe_val = curr_s, ts_comp, ts_val, te_comp, te_val
                        curr_s = 0
                
                # Catch the longest streak if it ended on the most recent comp
                if curr_s >= max_s and curr_s > 0:
                    max_s, fs_comp, fs_val, fe_comp, fe_val = curr_s, ts_comp, ts_val, te_comp, te_val
                
                if max_s > 0:
                    # Logic for the "Active" flame: Last comp must be a valid win AND part of the max streak
                    last_row = group.iloc[-1]
                    last_res = last_row['average'] if last_row['average'] > 0 else last_row['best']
                    is_active = (last_row['pos'] == 1 and last_res > 0 and curr_s == max_s)
                    
                    streaks.append({
                        'person_id': pid, 
                        'total': max_s, 
                        'streak_start': fs_comp, 
                        'streak_end': fe_comp, 
                        'is_active': is_active
                    })
            
            if streaks:
                # Merge with names and sort by total wins
                res = pd.DataFrame(streaks).merge(ph_names, on='person_id').sort_values(['total', 'person_id'], ascending=[False, True])
                for _, r in res.iterrows():
                    leaderboard.append({
                        'wca_id': r['person_id'], 
                        'name': r['person_name'], 
                        'total': int(r['total']), 
                        'streak_start': r['streak_start'], 
                        'streak_end': r['streak_end'], 
                        'is_active': r['is_active'], 
                        'ranks': {event_id: int(r['total'])}
                    })
    elif rank_type == 'podium':
        pods = ph_podiums_raw[ph_podiums_raw['event_id'].isin(selected_events)]
        if not pods.empty:
            tally = pods.groupby(['person_id', 'pos']).size().unstack(fill_value=0)
            for p in [1, 2, 3]: 
                if p not in tally.columns: tally[p] = 0
            tally['total'] = tally[1] + tally[2] + tally[3]
            res = tally.merge(ph_names, left_index=True, right_on='person_id').sort_values(['total', 1, 2, 3], ascending=False)
            for _, row in res.iterrows():
                person_pods = pods[pods['person_id'] == row['person_id']]
                ev_data = person_pods.groupby('event_id').size().to_dict()
                leaderboard.append({
                    'wca_id': row['person_id'], 'name': row['person_name'], 'total': int(row['total']),
                    'gold': int(row[1]), 'silver': int(row[2]), 'bronze': int(row[3]),
                    'ranks': {ev: int(ev_data.get(ev, 0)) for ev in selected_events}
                })

    elif rank_type == 'level':
        level_matrix = pd.DataFrame(0.0, index=ph_names['person_id'].values, columns=selected_events)
        for ev in selected_events:
            data_source = s_ranks if ev in ['333bf', '444bf', '555bf', '333mbf'] else a_ranks
            if data_source.empty or 'event_id' not in data_source.columns: continue
            global_ev = data_source[data_source['event_id'] == ev].set_index('person_id')['best'].dropna()
            if global_ev.empty: continue
            b = {100: global_ev.min(), 95: global_ev.quantile(0.001), 90: global_ev.quantile(0.01), 80: global_ev.quantile(0.05), 70: global_ev.quantile(0.10), 60: global_ev.quantile(0.20), 50: global_ev.quantile(0.50)}
            def get_s(v):
                if pd.isna(v) or v <= 0: return 0
                for s in [100, 95, 90, 80, 70, 60, 50]:
                    if v <= b[s]: return s
                return 40
            level_matrix.update(global_ev[global_ev.index.isin(ph_names['person_id'])].apply(get_s).to_frame(name=ev))
        level_matrix['total'] = level_matrix[selected_events].mean(axis=1)
        res = level_matrix[level_matrix['total'] > 0].merge(ph_names, left_index=True, right_on='person_id').sort_values('total', ascending=False)
        for _, r in res.iterrows():
            leaderboard.append({'wca_id': r['person_id'], 'name': r['person_name'], 'total': round(r['total'], 2), 'ranks': {ev: int(r[ev]) for ev in selected_events}})

    elif rank_type == 'kinch':
        if not s_ranks.empty and not a_ranks.empty:
            ph_s = s_ranks.set_index('person_id'); ph_a = a_ranks.set_index('person_id')
            nr_s = ph_s[ph_s['country_rank'] == 1].reset_index().set_index('event_id')['best']
            nr_a = ph_a[ph_a['country_rank'] == 1].reset_index().set_index('event_id')['best']
            p_s = ph_s.reset_index().pivot_table(index='person_id', columns='event_id', values='best').reindex(columns=selected_events)
            p_a = ph_a.reset_index().pivot_table(index='person_id', columns='event_id', values='best').reindex(columns=selected_events)
            km = pd.DataFrame(0.0, index=p_s.index, columns=selected_events)
            for ev in selected_events:
                if ev in ['333bf', '444bf', '555bf', '333fm']:
                    km[ev] = np.maximum((nr_s.get(ev, 0)/p_s[ev]*100).fillna(0), (nr_a.get(ev, 0)/p_a[ev]*100).fillna(0))
                elif ev == '333mbf':
                    if nr_s.get(ev, 0) > 0:
                        km[ev] = (decode_mbld_vectorized(p_s[ev])/decode_mbld_vectorized(pd.Series([nr_s[ev]])).iloc[0]*100).clip(lower=0).fillna(0)
                else:
                    if nr_a.get(ev, 0) > 0: km[ev] = (nr_a[ev]/p_a[ev]*100).fillna(0)
            km['total'] = km[selected_events].mean(axis=1)
            res = km[km['total'] > 0].merge(ph_names, left_index=True, right_on='person_id').sort_values('total', ascending=False)
            for _, r in res.iterrows():
                leaderboard.append({'wca_id': r['person_id'], 'name': r['person_name'], 'total': round(r['total'], 2), 'ranks': {ev: round(r[ev], 2) for ev in selected_events}})

    else: # single or average
        df = s_ranks if rank_type == 'single' else a_ranks
        if not df.empty and 'event_id' in df.columns:
            ph_filtered = df[(df['event_id'].isin(selected_events)) & (df['person_id'].isin(ph_names['person_id']))]
            if not ph_filtered.empty:
                max_r = ph_filtered.groupby('event_id')['country_rank'].max().to_dict()
                piv = ph_filtered.pivot(index='person_id', columns='event_id', values='country_rank').dropna(how='all')
                for ev in selected_events: piv[ev] = piv[ev].fillna(max_r.get(ev, 0) + 1)
                piv['total'] = piv[selected_events].sum(axis=1)
                res = piv.merge(ph_names, left_index=True, right_on='person_id').sort_values('total')
                for _, r in res.iterrows():
                    leaderboard.append({'wca_id': r['person_id'], 'name': r['person_name'], 'total': int(r['total']), 'ranks': {ev: int(r[ev]) for ev in selected_events}})

    total_count = len(leaderboard)
    total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
    return render_template('index.html', events=EVENTS, leaderboard=leaderboard[(page-1)*per_page:page*per_page], selected=selected_events, type=rank_type, updated=last_update_date, page=page, total_pages=total_pages)
ph_names, s_ranks, a_ranks, last_update_date, ph_podiums_raw, ph_all_results = load_and_cache()
if __name__ == '__main__':
    app.run(port=5000, debug=True)