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

def decode_mbld_vectorized(series):
    valid_mask = series.notna() & (series > 0)
    scores = pd.Series(0.0, index=series.index)
    if not valid_mask.any(): return scores
    s = series[valid_mask].astype(np.int64).astype(str).str.zfill(10)
    solved = 99 - s.str[1:3].astype(int) + s.str[8:10].astype(int)
    missed = s.str[8:10].astype(int)
    time_sec = s.str[3:8].astype(int)
    scores[valid_mask] = (solved - missed) + ((3600 - time_sec) / 3600)
    return scores

def load_and_cache():
    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = msgpack.unpackb(f.read(), raw=False)
                return pd.DataFrame(data['names']), pd.DataFrame(data['single']), pd.DataFrame(data['average']), data.get('last_update', 'Unknown'), pd.DataFrame(data.get('podiums', []))
        except: pass

    try:
        export_date = datetime.fromtimestamp(os.path.getmtime("WCA_export_results.tsv")).strftime('%b-%d, %Y')
        res_df = pd.read_csv("WCA_export_results.tsv", sep="\t", 
                             usecols=['person_id', 'person_name', 'person_country_id', 'pos', 'round_type_id', 'event_id', 'best', 'average', 'format_id'], 
                             low_memory=False)
        
        ph_res_all = res_df[res_df['person_country_id'] == 'Philippines']
        ph_names = ph_res_all.drop_duplicates('person_id')[['person_id', 'person_name']]
        ph_ids = ph_names['person_id'].unique()

        # --- REVISED PODIUM FILTER ---
        pod_filter = ph_res_all[
            (ph_res_all['pos'] <= 3) & 
            (ph_res_all['round_type_id'].str.lower().isin(['f', 'c']))
        ].copy()

        def is_valid_podium(row):
            ev = row['event_id']
            fmt = row['format_id'] 
            # DNF is -1, DNS is -2. Any valid result must be > 0.
            
            # 1. Blindfolded events are always ranked by Single
            if ev in ['333bf', '444bf', '555bf', '333mbf']:
                return row['best'] > 0
            
            # 2. FMC logic
            if ev == '333fm':
                if fmt == 'm': # Mean of 3 format
                    return row['average'] > 0
                return row['best'] > 0 # Best of X format
            
            # 3. Standard events
            # If the format is 'a' (Avg of 5) or 'm' (Mean of 3), the podium is based on Average
            if fmt in ['a', 'm']:
                return row['average'] > 0
            
            # 4. Fallback for Best of X rounds
            return row['best'] > 0

        ph_podiums = pod_filter[pod_filter.apply(is_valid_podium, axis=1)]

        s_ranks = pd.read_csv("WCA_export_ranks_single.tsv", sep="\t", usecols=['person_id', 'event_id', 'best', 'country_rank'], low_memory=False)
        a_ranks = pd.read_csv("WCA_export_ranks_average.tsv", sep="\t", usecols=['person_id', 'event_id', 'best', 'country_rank'], low_memory=False)
        
        ph_s_ranks = s_ranks[s_ranks['person_id'].isin(ph_ids)]
        ph_a_ranks = a_ranks[a_ranks['person_id'].isin(ph_ids)]

        cache_data = {
            'names': ph_names.to_dict('records'),
            'single': ph_s_ranks.to_dict('records'),
            'average': ph_a_ranks.to_dict('records'),
            'podiums': ph_podiums[['person_id', 'event_id', 'pos']].to_dict('records'),
            'last_update': export_date
        }
        with open(CACHE_FILE, 'wb') as f:
            f.write(msgpack.packb(cache_data))

        return ph_names, ph_s_ranks, ph_a_ranks, export_date, ph_podiums
    except Exception as e:
        return None, None, None, "Unknown", pd.DataFrame()

ph_names, s_ranks, a_ranks, last_update_date, ph_podiums_raw = load_and_cache()

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

    if rank_type == 'podium':
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
    else:
        df = s_ranks if rank_type == 'single' else a_ranks
        ph_filtered = df[(df['event_id'].isin(selected_events)) & (df['person_id'].isin(ph_names['person_id']))]
        if not ph_filtered.empty:
            max_r = ph_filtered.groupby('event_id')['country_rank'].max().to_dict()
            piv = ph_filtered.pivot(index='person_id', columns='event_id', values='country_rank').dropna(how='all')
            for ev in selected_events: piv[ev] = piv[ev].fillna(max_r.get(ev, 0) + 1)
            piv['total'] = piv[selected_events].sum(axis=1)
            res = piv.merge(ph_names, left_index=True, right_on='person_id').sort_values('total')
            for _, r in res.iterrows():
                leaderboard.append({'wca_id': r['person_id'], 'name': r['person_name'], 'total': int(r['total']), 'ranks': {ev: int(r[ev]) for ev in selected_events}})

    total_pages = math.ceil(len(leaderboard) / per_page)
    return render_template('index.html', events=EVENTS, leaderboard=leaderboard[(page-1)*per_page:page*per_page], selected=selected_events, type=rank_type, updated=last_update_date, page=page, total_pages=total_pages)

if __name__ == '__main__':
    app.run(port=5000, debug=True)