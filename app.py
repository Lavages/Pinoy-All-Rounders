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
    """Vectorized decoding of MBLD results into points."""
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
        print("🚀 Loading Rankings from Cache...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = msgpack.unpackb(f.read(), raw=False)
                # Reconstruct DataFrames from the slimmed dictionaries
                s_df = pd.DataFrame(data['single']).set_index('person_id')
                a_df = pd.DataFrame(data['average']).set_index('person_id')
                return pd.DataFrame(data['names']), s_df, a_df, data.get('last_update', 'Unknown')
        except Exception as e:
            print(f"⚠️ Cache read failed: {e}")

    print("📦 Processing TSVs...")
    try:
        # 1. Get Export Date
        export_date = datetime.fromtimestamp(os.path.getmtime("WCA_export_results.tsv")).strftime('%b-%d, %Y')
        
        # 2. Process Names (Only Philippines)
        res_df = pd.read_csv("WCA_export_results.tsv", sep="\t", usecols=['person_id', 'person_name', 'person_country_id'], low_memory=False)
        ph_names = res_df[res_df['person_country_id'] == 'Philippines'].drop_duplicates('person_id')[['person_id', 'person_name']]
        ph_ids = ph_names['person_id'].unique()

        # 3. Process Ranks (Only Philippines & Only necessary columns)
        cols_to_keep = ['person_id', 'event_id', 'best', 'country_rank']
        
        s_ranks = pd.read_csv("WCA_export_ranks_single.tsv", sep="\t", usecols=cols_to_keep, low_memory=False)
        a_ranks = pd.read_csv("WCA_export_ranks_average.tsv", sep="\t", usecols=cols_to_keep, low_memory=False)
        
        # CRITICAL OPTIMIZATION: Filter for PH IDs before saving to cache
        # This reduces the rows from millions (Global) to thousands (PH Only)
        ph_s_ranks = s_ranks[s_ranks['person_id'].isin(ph_ids)]
        ph_a_ranks = a_ranks[a_ranks['person_id'].isin(ph_ids)]

        cache_data = {
            'names': ph_names.to_dict(orient='records'),
            'single': ph_s_ranks.to_dict(orient='records'),
            'average': ph_a_ranks.to_dict(orient='records'),
            'last_update': export_date
        }
        
        with open(CACHE_FILE, 'wb') as f:
            f.write(msgpack.packb(cache_data))

        print(f"✅ Cache Created: {CACHE_FILE}")
        return ph_names, ph_s_ranks.set_index('person_id'), ph_a_ranks.set_index('person_id'), export_date

    except Exception as e:
        print(f"❌ Error during load: {e}")
        return None, None, None, "Unknown"

ph_names, s_ranks, a_ranks, last_update_date = load_and_cache()

@app.route('/', methods=['GET', 'POST'])
def index():
    # Support both POST (form) and GET (pagination links)
    if request.method == 'POST':
        selected_events = request.form.getlist('events')
        rank_type = request.form.get('rank_type', 'single')
    else:
        selected_events = request.args.getlist('events')
        rank_type = request.args.get('rank_type', 'single')

    if not selected_events: selected_events = ['333']
    
    page = int(request.args.get('page', 1))
    per_page = 50
    leaderboard = []

    if ph_names is None: return "Data Load Error."
    ph_ids = ph_names['person_id'].values

    # --- LOGIC: LEVEL ---
    if rank_type == 'level':
        # Initialize matrix with 0.0 for all PH cubers and selected events
        level_matrix = pd.DataFrame(0.0, index=ph_ids, columns=selected_events)
        
        for ev in selected_events:
            is_bld = ev in ['333bf', '444bf', '555bf', '333mbf']
            data_source = s_ranks if is_bld else a_ranks
            global_ev = data_source[data_source['event_id'] == ev]['best'].dropna()
            
            if global_ev.empty: continue
            
            boundaries = {
                100: global_ev.min(),
                95: global_ev.quantile(0.001),
                90: global_ev.quantile(0.01),
                80: global_ev.quantile(0.05),
                70: global_ev.quantile(0.10),
                60: global_ev.quantile(0.20),
                50: global_ev.quantile(0.50)
            }
            
            def get_score(val):
                if pd.isna(val) or val <= 0: return 0
                if val <= boundaries[100]: return 100
                if val <= boundaries[95]: return 95
                if val <= boundaries[90]: return 90
                if val <= boundaries[80]: return 80
                if val <= boundaries[70]: return 70
                if val <= boundaries[60]: return 60
                if val <= boundaries[50]: return 50
                return 40

            # Get scores for PH cubers and update the matrix
            ph_results = global_ev[global_ev.index.isin(ph_ids)]
            event_scores = ph_results.apply(get_score)
            level_matrix.update(event_scores.to_frame(name=ev))

        level_matrix['total'] = level_matrix[selected_events].mean(axis=1)
        res = level_matrix[level_matrix['total'] > 0].merge(ph_names, left_index=True, right_on='person_id')
        res = res.sort_values('total', ascending=False)
        
        for _, row in res.iterrows():
            leaderboard.append({
                'wca_id': row['person_id'], 'name': row['person_name'], 'total': round(row['total'], 2),
                'ranks': {ev: int(float(row[ev])) for ev in selected_events}
            })

    elif rank_type == 'kinch':
        # 1. Filter for Filipinos and select events
        ph_s = s_ranks[s_ranks.index.isin(ph_ids)]
        ph_a = a_ranks[a_ranks.index.isin(ph_ids)]

        # 2. Identify the NR (National Record) value for each event
        # We find the 'best' time/score where country_rank is 1
        nr_s = ph_s[ph_s['country_rank'] == 1].set_index('event_id')['best']
        nr_a = ph_a[ph_a['country_rank'] == 1].set_index('event_id')['best']
        
        # 3. Pivot Personal Bests (PBs) for all PH cubers
        p_s = ph_s.pivot_table(index='person_id', columns='event_id', values='best').reindex(columns=selected_events)
        p_a = ph_a.pivot_table(index='person_id', columns='event_id', values='best').reindex(columns=selected_events)
        
        kinch_matrix = pd.DataFrame(0.0, index=p_s.index, columns=selected_events)
        
        for ev in selected_events:
            # RULE: 3BLD, 4BLD, 5BLD, and FMC (Better of Single or Average)
            if ev in ['333bf', '444bf', '555bf', '333fm']:
                val_nr_s = nr_s.get(ev, 0)
                val_nr_a = nr_a.get(ev, 0)
                
                score_s = (val_nr_s / p_s[ev] * 100).fillna(0)
                score_a = (val_nr_a / p_a[ev] * 100).fillna(0)
                kinch_matrix[ev] = np.maximum(score_s, score_a)
                
            # RULE: MBLD (Points + Time Decimal from Single)
            elif ev == '333mbf':
                val_nr_mbf = nr_s.get(ev, 0)
                if val_nr_mbf > 0:
                    nr_pts = decode_mbld_vectorized(pd.Series([val_nr_mbf])).iloc[0]
                    player_pts = decode_mbld_vectorized(p_s[ev])
                    kinch_matrix[ev] = (player_pts / nr_pts * 100).clip(lower=0).fillna(0)
            
            # RULE: Standard Events (Average Only)
            else:
                val_nr_a = nr_a.get(ev, 0)
                if val_nr_a > 0:
                    kinch_matrix[ev] = (val_nr_a / p_a[ev] * 100).fillna(0)

        # 4. Final Average across all selected events
        kinch_matrix['total'] = kinch_matrix[selected_events].mean(axis=1)
        
        # Merge with names and sort descending
        res = kinch_matrix[kinch_matrix['total'] > 0].merge(ph_names, left_index=True, right_on='person_id')
        res = res.sort_values('total', ascending=False)
        
        for _, row in res.iterrows():
            leaderboard.append({
                'wca_id': row['person_id'], 
                'name': row['person_name'], 
                'total': round(row['total'], 2), 
                'ranks': {ev: round(row[ev], 2) for ev in selected_events}
            })
    else:
        df = s_ranks if rank_type == 'single' else a_ranks
        ph_filtered = df[(df['event_id'].isin(selected_events)) & (df.index.isin(ph_ids))].reset_index()
        if not ph_filtered.empty:
            max_ranks = ph_filtered.groupby('event_id')['country_rank'].max().to_dict()
            pivot = ph_filtered.pivot(index='person_id', columns='event_id', values='country_rank')
            
            # Drop people who haven't competed in ANY of the selected events
            pivot = pivot.dropna(how='all')
            
            for ev in selected_events:
                penalty = max_ranks.get(ev, 0) + 1
                pivot[ev] = pivot[ev].fillna(penalty)
            
            pivot['total'] = pivot[selected_events].sum(axis=1)
            
            # CRITICAL: Filter out anyone with 0 total (though pivot.dropna(how='all') mostly covers this)
            res = pivot[pivot['total'] > 0].merge(ph_names, left_index=True, right_on='person_id').sort_values('total')
            
            for _, row in res.iterrows():
                leaderboard.append({
                    'wca_id': row['person_id'], 
                    'name': row['person_name'], 
                    'total': int(row['total']), 
                    'ranks': {ev: int(row[ev]) for ev in selected_events}, 
                    'max_ranks': max_ranks
                })

    # Pagination
    total_items = len(leaderboard)
    total_pages = math.ceil(total_items / per_page)
    paged_leaderboard = leaderboard[(page-1)*per_page : page*per_page]

    return render_template('index.html', 
                           events=EVENTS, 
                           leaderboard=paged_leaderboard, 
                           selected=selected_events, 
                           type=rank_type, 
                           updated=last_update_date,
                           page=page,
                           total_pages=total_pages)

if __name__ == '__main__':
    app.run(port=5000, debug=True)