from flask import Flask, request, render_template, jsonify
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

def format_wca_time(value, event_id, is_avg=False):
    """
    Formats WCA integers into readable strings based on event type.
    - Normal: M:SS.hh or SS.hh
    - FMC: Single as Int, Average as Decimal (xx.xx)
    - MBLD: Solved/Attempted M:SS
    """
    if value <= 0 or pd.isna(value): 
        return "DNF" if value == -1 else "-"
    
    # 1. FMC (Fewest Moves)
    if event_id == '333fm': 
        return f"{value/100:.2f}" if is_avg else str(int(value))
    
    # 2. Multi-Blind (333mbf) -> Solved/Attempted M:SS
    if event_id == '333mbf':
        s = str(int(value)).zfill(10)
        # 0DDTTTTTMM -> DD: Difference, TTTTT: Time in Sec, MM: Missed
        dd = int(s[1:3])
        time_sec = int(s[3:8])
        mm = int(s[8:10])
        
        difference = 99 - dd
        solved = difference + mm
        attempted = solved + mm
        
        if time_sec == 99999:
            return f"{solved}/{attempted} ?:??"
            
        minutes = time_sec // 60
        seconds = time_sec % 60
        return f"{solved}/{attempted} {minutes}:{str(seconds).zfill(2)}"
    
    # 3. Standard Time Formatting (Centiseconds to Clock)
    val = int(value)
    cs = val % 100
    total_sec = val // 100
    s = total_sec % 60
    m = (total_sec // 60) % 60
    h = total_sec // 3600

    if h > 0:
        return f"{h}:{str(m).zfill(2)}:{str(s).zfill(2)}.{str(cs).zfill(2)}"
    elif m > 0:
        return f"{m}:{str(s).zfill(2)}.{str(cs).zfill(2)}"
    else:
        # Handles 0.69 case
        return f"{s}.{str(cs).zfill(2)}"

def decode_mbld_vectorized(series):
    valid_mask = series.notna() & (series > 0)
    scores = pd.Series(0.0, index=series.index)
    if not valid_mask.any(): return scores
    
    # Ensure 10 digits: 0DDTTTTTMM
    s = series[valid_mask].astype(np.int64).astype(str).str.zfill(10)
    
    # difference = 99 - DD
    difference = 99 - s.str[1:3].astype(int)
    # TTTTT = time in seconds
    time_sec = s.str[3:8].astype(int)
    
    # To rank correctly: Points are primary, Time is secondary (tie-breaker)
    # We use (Difference) + (Remaining time ratio)
    # This ensures a better performance results in a HIGHER decoded score.
    scores[valid_mask] = difference + ((3600 - time_sec) / 3600)
    return scores

def load_and_cache():
    # Initialize empty defaults to prevent "UnboundLocalError"
    ph_names = pd.DataFrame()
    ph_s_ranks = pd.DataFrame()
    ph_a_ranks = pd.DataFrame()
    ph_podiums = pd.DataFrame()
    ph_all_results = pd.DataFrame()
    export_date = "Unknown"

    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = msgpack.unpackb(f.read(), raw=False)
                df_all = pd.DataFrame(data.get('all_results', []))
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

    try:
        required_files = ["WCA_export_results.tsv", "WCA_export_ranks_single.tsv", 
                          "WCA_export_ranks_average.tsv", "WCA_export_competitions.tsv"]
        
        for f in required_files:
            if not os.path.exists(f):
                return ph_names, ph_s_ranks, ph_a_ranks, "Error: Missing Files", ph_podiums, ph_all_results

        export_date = datetime.fromtimestamp(os.path.getmtime("WCA_export_results.tsv")).strftime('%b-%d, %Y')
        
        # Load Results
        res_df = pd.read_csv("WCA_export_results.tsv", sep="\t", usecols=['person_id', 'person_name', 'person_country_id', 'pos', 'round_type_id', 'event_id', 'best', 'average', 'competition_id'], low_memory=False)
        comps = pd.read_csv("WCA_export_competitions.tsv", sep="\t", usecols=['id', 'year', 'month', 'day', 'name'])
        comps['date'] = pd.to_datetime(comps[['year', 'month', 'day']])
        
        # Filter for Philippines
        ph_all_results = res_df[res_df['person_country_id'] == 'Philippines'].merge(comps[['id', 'date', 'name']], left_on='competition_id', right_on='id')
        ph_names = ph_all_results.drop_duplicates('person_id')[['person_id', 'person_name']]
        ph_ids = ph_names['person_id'].unique()
        
        # Load Ranks
        s_ranks_raw = pd.read_csv("WCA_export_ranks_single.tsv", sep="\t", usecols=['person_id', 'event_id', 'best', 'country_rank'], low_memory=False)
        a_ranks_raw = pd.read_csv("WCA_export_ranks_average.tsv", sep="\t", usecols=['person_id', 'event_id', 'best', 'country_rank'], low_memory=False)
        ph_s_ranks = s_ranks_raw[s_ranks_raw['person_id'].isin(ph_ids)]
        ph_a_ranks = a_ranks_raw[a_ranks_raw['person_id'].isin(ph_ids)]
        
        # Prepare for cache
        ph_res_serializable = ph_all_results.copy()
        ph_res_serializable['date'] = ph_res_serializable['date'].dt.strftime('%Y-%m-%d')
        
        ph_podiums = ph_all_results[(ph_all_results['pos'] <= 3) & (ph_all_results['round_type_id'].isin(['f', 'c']))][['person_id', 'event_id', 'pos']]

        cache_data = {
            'names': ph_names.to_dict('records'), 
            'single': ph_s_ranks.to_dict('records'),
            'average': ph_a_ranks.to_dict('records'), 
            'podiums': ph_podiums.to_dict('records'),
            'all_results': ph_res_serializable[['person_id', 'event_id', 'pos', 'round_type_id', 'date', 'name', 'best', 'average']].to_dict('records'),
            'last_update': export_date
        }
        
        with open(CACHE_FILE, 'wb') as f: 
            f.write(msgpack.packb(cache_data))
            
        return ph_names, ph_s_ranks, ph_a_ranks, export_date, ph_podiums, ph_all_results

    except Exception as e:
        print(f"Data processing error: {e}")
        return ph_names, ph_s_ranks, ph_a_ranks, "Error", ph_podiums, ph_all_results

# Global variables assignment at the bottom of the script
ph_names, s_ranks, a_ranks, last_update_date, ph_podiums_raw, ph_all_results = load_and_cache()

@app.route('/battle', methods=['GET', 'POST'])
def battle():
    c1_id = request.args.get('c1', '')
    c2_id = request.args.get('c2', '')
    
    comp1, comp2 = None, None
    battle_data = []
    kinch_scores = {'c1': {}, 'c2': {}}
    summary = {'c1_wins': 0, 'c2_wins': 0, 'c1_total_kinch': 0, 'c2_total_kinch': 0}

    if c1_id and c2_id:
        # Resolve Names/IDs
        def find_person(query):
            match = ph_names[(ph_names['person_id'] == query) | (ph_names['person_name'].str.contains(query, case=False, na=False))]
            return match.iloc[0].to_dict() if not match.empty else None

        comp1 = find_person(c1_id)
        comp2 = find_person(c2_id)

        if comp1 and comp2:
            p1_id, p2_id = comp1['person_id'], comp2['person_id']
            
            # Get NR for Kinch calculation
            nr_s = s_ranks[s_ranks['country_rank'] == 1].set_index('event_id')['best']
            nr_a = a_ranks[a_ranks['country_rank'] == 1].set_index('event_id')['best']

            for ev_id, ev_name in EVENTS.items():
                # Get Rankings
                r1_s = s_ranks[(s_ranks['person_id'] == p1_id) & (s_ranks['event_id'] == ev_id)]
                r2_s = s_ranks[(s_ranks['person_id'] == p2_id) & (s_ranks['event_id'] == ev_id)]
                r1_a = a_ranks[(a_ranks['person_id'] == p1_id) & (a_ranks['event_id'] == ev_id)]
                r2_a = a_ranks[(a_ranks['person_id'] == p2_id) & (a_ranks['event_id'] == ev_id)]

                v1_s = r1_s['best'].iloc[0] if not r1_s.empty else 0
                v2_s = r2_s['best'].iloc[0] if not r2_s.empty else 0
                v1_a = r1_a['best'].iloc[0] if not r1_a.empty else 0
                v2_a = r2_a['best'].iloc[0] if not r2_a.empty else 0

                # Kinch Logic
                def get_kinch(ev, val_s, val_a):
                    if ev == '333mbf':
                        return decode_mbld_vectorized(pd.Series([val_s])).iloc[0] / decode_mbld_vectorized(pd.Series([nr_s.get(ev, 0)])).iloc[0] * 100 if val_s > 0 else 0
                    if ev in ['333bf', '444bf', '555bf', '333fm']:
                        score = 0
                        if val_s > 0: score = max(score, (nr_s.get(ev, 0) / val_s * 100))
                        if val_a > 0: score = max(score, (nr_a.get(ev, 0) / val_a * 100))
                        return score
                    return (nr_a.get(ev, 0) / val_a * 100) if val_a > 0 else 0

                k1 = round(get_kinch(ev_id, v1_s, v1_a), 2)
                k2 = round(get_kinch(ev_id, v2_s, v2_a), 2)
                kinch_scores['c1'][ev_name] = k1
                kinch_scores['c2'][ev_name] = k2

                # Comparison Logic (lower is better, except MBLD)
                def compare(val1, val2, is_mbld=False):
                    if val1 <= 0: return 'c2' if val2 > 0 else None
                    if val2 <= 0: return 'c1'
                    if is_mbld: return 'c1' if val1 < val2 else 'c2' # WCA MBLD storage: lower is better
                    return 'c1' if val1 < val2 else 'c2' if val2 < val1 else None

                win_s = compare(v1_s, v2_s, ev_id == '333mbf')
                win_a = compare(v1_a, v2_a)
                if win_s == 'c1': summary['c1_wins'] += 1
                elif win_s == 'c2': summary['c2_wins'] += 1
                if win_a == 'c1': summary['c1_wins'] += 1
                elif win_a == 'c2': summary['c2_wins'] += 1

                battle_data.append({
                    'event': ev_name,
                    'c1_s': format_wca_time(v1_s, ev_id), 'c2_s': format_wca_time(v2_s, ev_id),
                    'c1_a': format_wca_time(v1_a, ev_id), 'c2_a': format_wca_time(v2_a, ev_id),
                    'win_s': win_s, 'win_a': win_a,
                    'rank1_s': int(r1_s['country_rank'].iloc[0]) if not r1_s.empty else None,
                    'rank2_s': int(r2_s['country_rank'].iloc[0]) if not r2_s.empty else None
                })
            
            summary['c1_total_kinch'] = round(sum(kinch_scores['c1'].values()) / len(EVENTS), 2)
            summary['c2_total_kinch'] = round(sum(kinch_scores['c2'].values()) / len(EVENTS), 2)

    return render_template('battle.html', comp1=comp1, comp2=comp2, battle_data=battle_data, 
                           kinch=kinch_scores, summary=summary, updated=last_update_date)

@app.route('/best-of-ph', methods=['GET'])
def best_of_ph():
    query = request.args.get('query', '')
    results = []
    person = None

    if query:
        match = ph_names[(ph_names['person_id'] == query) | 
                         (ph_names['person_name'].str.contains(query, case=False, na=False))]
        
        if not match.empty:
            person = match.iloc[0].to_dict()
            p_id, p_name = person['person_id'], person['person_name']
            
            # --- PREPARE DATA ---
            p_singles = s_ranks[s_ranks['person_id'] == p_id]
            p_averages = a_ranks[a_ranks['person_id'] == p_id]
            p_results = ph_all_results[ph_all_results['person_id'] == p_id].sort_values('date', ascending=False)
            
            first_name = p_name.split()[0]
            debut_year = p_id[:4]
            fn_ids = ph_names[ph_names['person_name'].str.startswith(first_name)]['person_id']
            year_ids = ph_names[ph_names['person_id'].str.startswith(debut_year)]['person_id']

            # --- PROCESS BOTH SINGLES & AVERAGES ---
            datasets = [
                (p_singles, s_ranks, "Single", False),
                (p_averages, a_ranks, "Average", True)
            ]

            for p_data, global_ranks, label, is_avg in datasets:
                for _, row in p_data.iterrows():
                    eid = row['event_id']
                    event_name = EVENTS.get(eid, eid)
                    formatted_time = format_wca_time(row['best'], eid, is_avg)
                    
                    # 1. National Best Check
                    if row['country_rank'] == 1:
                        results.append({
                            "cat": "National Best", 
                            "text": f"Current PH National Record: {event_name} {label} ({formatted_time})"
                        })

                    # 2. Name-Based Check (Fastest among others with same first name)
                    if len(fn_ids) > 1:
                        best_in_name = global_ranks[global_ranks['person_id'].isin(fn_ids) & (global_ranks['event_id'] == eid)]['best'].min()
                        if row['best'] == best_in_name:
                            results.append({
                                "cat": "Name", 
                                "text": f"Fastest {event_name} {label} ({formatted_time}) among Filipinos named '{first_name}'"
                            })

                    # 3. WCA ID / Debut Year Check
                    best_in_year = global_ranks[global_ranks['person_id'].isin(year_ids) & (global_ranks['event_id'] == eid)]['best'].min()
                    if row['best'] == best_in_year:
                        results.append({
                            "cat": "WCA ID", 
                            "text": f"Fastest {event_name} {label} ({formatted_time}) in the 'Class of {debut_year}'"
                        })

            # --- CATEGORY: TIME (Decimal Matching) ---
            for _, row in p_singles.iterrows():
                eid = row['event_id']
                if eid in ['333fm', '333mbf']: continue 
                
                val_str = f"{row['best']/100:.2f}"
                decimal = val_str.split('.')[-1]
                same_decimal_best = s_ranks[(s_ranks['event_id'] == eid) & 
                                            ((s_ranks['best']/100).astype(str).str.endswith(decimal))]['best'].min()
                
                if row['best'] == same_decimal_best:
                    results.append({
                        "cat": "Time", 
                        "text": f"Fastest {EVENTS.get(eid)} among those with a '.{decimal}' PB"
                    })

            # --- CATEGORY: UNIQUE & COMPETITION ---
            if len(ph_names[ph_names['person_name'] == p_name]) == 1:
                results.append({"cat": "Unique", "text": f"The only '{p_name}' in Philippine Speedcubing"})

            if not p_results.empty:
                comp_col = next((c for c in ['competition_id', 'competitionId', 'id'] if c in p_results.columns), None)
                if comp_col:
                    last_comp_id = p_results.iloc[0][comp_col]
                    last_comp_name = p_results.iloc[0]['name']
                    comp_res = ph_all_results[ph_all_results[comp_col] == last_comp_id]
                    comp_bests = comp_res.groupby('event_id')['best'].min()
                    
                    for _, row in p_results[p_results[comp_col] == last_comp_id].iterrows():
                        if row['best'] == comp_bests.get(row['event_id']):
                            f_time = format_wca_time(row['best'], row['event_id'], False)
                            results.append({
                                "cat": "Competition", 
                                "text": f"Fastest {EVENTS.get(row['event_id'])} ({f_time}) at {last_comp_name}"
                            })

    return render_template('best_of_ph.html', person=person, results=results)

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
            df_ev = ph_all_results[(ph_all_results['event_id'] == event_id) & 
                                   (ph_all_results['round_type_id'].isin(['f','c']))].sort_values(['person_id', 'date'])
            streaks = []
            for pid, group in df_ev.groupby('person_id'):
                max_s, curr_s = 0, 0
                ts_comp, ts_val, te_comp, te_val = "", "", "", ""
                fs_comp, fs_val, fe_comp, fe_val = "", "", "", ""
                for _, row in group.iterrows():
                    res_value = row['average'] if row['average'] > 0 else row['best']
                    if row['pos'] == 1 and res_value > 0:
                        val_str = format_wca_time(res_value, event_id)
                        if curr_s == 0: ts_comp, ts_val = row['name'], val_str
                        curr_s += 1
                        te_comp, te_val = row['name'], val_str
                    else:
                        if curr_s >= max_s and curr_s > 0:
                            max_s, fs_comp, fs_val, fe_comp, fe_val = curr_s, ts_comp, ts_val, te_comp, te_val
                        curr_s = 0
                if curr_s >= max_s and curr_s > 0:
                    max_s, fs_comp, fs_val, fe_comp, fe_val = curr_s, ts_comp, ts_val, te_comp, te_val
                if max_s > 0:
                    last_row = group.iloc[-1]
                    last_res = last_row['average'] if last_row['average'] > 0 else last_row['best']
                    is_active = (last_row['pos'] == 1 and last_res > 0 and curr_s == max_s)
                    streaks.append({'person_id': pid, 'total': max_s, 'streak_start': fs_comp, 'streak_end': fe_comp, 'is_active': is_active})
            if streaks:
                res = pd.DataFrame(streaks).merge(ph_names, on='person_id').sort_values(['total', 'person_id'], ascending=[False, True])
                for _, r in res.iterrows():
                    leaderboard.append({'wca_id': r['person_id'], 'name': r['person_name'], 'total': int(r['total']), 'is_active': r['is_active'], 'ranks': {event_id: int(r['total'])}})

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
                leaderboard.append({'wca_id': row['person_id'], 'name': row['person_name'], 'total': int(row['total']), 'gold': int(row[1]), 'silver': int(row[2]), 'bronze': int(row[3]), 'ranks': {ev: int(ev_data.get(ev, 0)) for ev in selected_events}})

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
app = app

if __name__ == '__main__':
    app.run(debug=True)