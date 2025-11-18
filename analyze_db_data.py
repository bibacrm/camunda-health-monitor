"""
Camunda 7 Fast Process Analysis Script
Console-only output, focuses on business-critical processes
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from helpers.db_helper import execute_query, init_db_pool, get_db_helper
import concurrent.futures

# ============================================================================
# CONFIGURATION
# ============================================================================

CATEGORY_DEFS = [
    ("ultra_fast", 5/60/60),          # <5 seconds - SKIPPED (internal only)
    ("very_fast", 0.5/60),            # <30 seconds
    ("fast_background", 0.1),         # <6 minutes
    ("standard", 0.5),                # <30 minutes
    ("extended", 4),                  # <4 hours
    ("long_running", 24),             # <24 hours
    ("batch_manual", float("inf")),   # 24h+
]

ANALYZE_CATEGORIES = ['very_fast', 'fast_background', 'standard', 'extended', 'long_running', 'batch_manual']

CATEGORY_LABELS = {
    "ultra_fast": "Ultra Fast (<5s)",
    "very_fast": "Very Fast (5-30s)",
    "fast_background": "Fast Background (<6m)",
    "standard": "Standard (6m-30m)",
    "extended": "Extended (30m-4h)",
    "long_running": "Long Running (4h-24h)",
    "batch_manual": "Batch / Manual (24h+)"
}

# ============================================================================
# UTILITIES
# ============================================================================

def safe_float(x):
    """Safely convert to float"""
    if x is None:
        return None
    return float(x)


def classify_duration_hours(median_hours: float) -> str:
    """Classify process into category"""
    for name, upper in CATEGORY_DEFS:
        if median_hours <= upper:
            return name
    return "batch_manual"


def compute_basic_stats(values: np.ndarray):
    """Compute statistics"""
    if len(values) == 0:
        return None

    stats_dict = {}
    stats_dict["count"] = int(len(values))
    stats_dict["mean_s"] = float(np.mean(values))
    stats_dict["median_s"] = float(np.median(values))
    stats_dict["std_s"] = float(np.std(values))
    stats_dict["min_s"] = float(np.min(values))
    stats_dict["max_s"] = float(np.max(values))

    for p in [50, 75, 90, 95, 99]:
        stats_dict[f"p{p}_s"] = float(np.percentile(values, p))

    if stats_dict["mean_s"] > 0:
        stats_dict["cv"] = round(stats_dict["std_s"] / stats_dict["mean_s"], 4)
    else:
        stats_dict["cv"] = None

    # Outliers
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    outlier_threshold = q3 + 3 * iqr
    outliers = values[values > outlier_threshold]

    stats_dict["iqr_s"] = float(iqr)
    stats_dict["outlier_threshold_s"] = float(outlier_threshold)
    stats_dict["outlier_count"] = int(len(outliers))
    stats_dict["outlier_pct"] = round(len(outliers) / len(values) * 100, 3) if len(values) else 0.0

    return stats_dict


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def categorize_all_processes(days: int = 180):
    """Phase 1: Quick categorization of ALL processes"""
    print("Phase 1: Quick categorization of all processes...")

    query = f"""
        SELECT 
            proc_def_key_,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as median_duration,
            COUNT(*) as instance_count
        FROM act_hi_procinst
        WHERE start_time_ > NOW() - INTERVAL '{days} days'
          AND end_time_ IS NOT NULL
          AND end_time_ > start_time_
        GROUP BY proc_def_key_
        HAVING COUNT(*) >= 5
    """

    results = execute_query(query)
    if not results:
        print("No data found")
        return {}

    process_categories = {}
    category_counts = {}

    for row in results:
        proc_key = row['proc_def_key_']
        median_seconds = safe_float(row['median_duration']) or 0
        median_hours = median_seconds / 3600.0
        category = classify_duration_hours(median_hours)

        process_categories[proc_key] = {
            'category': category,
            'median_seconds': median_seconds,
            'median_hours': median_hours,
            'instance_count': int(row.get('instance_count', 0))
        }
        category_counts[category] = category_counts.get(category, 0) + 1

    # Print summary
    print(f"\n✓ Categorized {len(process_categories)} processes:")
    for cat in ['ultra_fast', 'very_fast', 'fast_background', 'standard', 'extended', 'long_running', 'batch_manual']:
        count = category_counts.get(cat, 0)
        if count > 0:
            skip_marker = " (SKIPPED - internal only)" if cat == 'ultra_fast' else ""
            print(f"  {CATEGORY_LABELS[cat]}: {count} processes{skip_marker}")

    return process_categories


def analyze_business_critical_processes(process_categories: dict, days: int = 180):
    """Phase 2: Detailed analysis of business-critical only"""
    analyze_list = [k for k, v in process_categories.items() if v['category'] in ANALYZE_CATEGORIES]

    if not analyze_list:
        print("\n⚠️  No business-critical processes to analyze")
        return []

    print(f"\nPhase 2: Detailed analysis of {len(analyze_list)} business-critical processes...")

    proc_keys_str = "', '".join(analyze_list)

    query = f"""
        SELECT 
            proc_def_key_,
            SUBSTRING(proc_def_id_ FROM ':([0-9]+):') as version,
            EXTRACT(EPOCH FROM (end_time_ - start_time_)) as duration_seconds
        FROM act_hi_procinst
        WHERE start_time_ > NOW() - INTERVAL '{days} days'
          AND end_time_ IS NOT NULL
          AND end_time_ > start_time_
          AND proc_def_key_ IN ('{proc_keys_str}')
        ORDER BY proc_def_key_, end_time_
    """

    results = execute_query(query)
    if not results:
        return []

    df = pd.DataFrame(results)
    if 'duration_seconds' in df.columns:
        df['duration_seconds'] = df['duration_seconds'].apply(safe_float)

    per_process_summary = []

    for proc_key in analyze_list[:50]:
        proc_df = df[df['proc_def_key_'] == proc_key].copy()
        durations = proc_df['duration_seconds'].values
        durations = durations[durations > 0]

        if len(durations) == 0:
            continue

        stats_dict = compute_basic_stats(durations)
        if not stats_dict:
            continue

        category = process_categories[proc_key]['category']

        record = {
            'process_key': proc_key,
            'category': category,
            'stats': stats_dict,
        }
        per_process_summary.append(record)

    print(f"✓ Analyzed {len(per_process_summary)} processes\n")

    return per_process_summary


def print_detailed_analysis(per_process_summary: list):
    """Print detailed analysis"""

    by_category = {}
    for p in per_process_summary:
        cat = p['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p)

    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS BY CATEGORY")
    print(f"{'='*80}\n")

    for cat in ANALYZE_CATEGORIES:
        if cat not in by_category:
            continue

        processes = by_category[cat]

        print(f"\n{CATEGORY_LABELS[cat]}: {len(processes)} processes")
        print("─" * 80)

        for p in processes[:10]:
            s = p['stats']

            print(f"  {p['process_key']}")
            print(f"    Median: {s['median_s']:.2f}s ({s['median_s']/60:.2f}m) | P95: {s['p95_s']:.2f}s | P99: {s['p99_s']:.2f}s")
            print(f"    CV: {s.get('cv', 'N/A')} | Outliers: {s['outlier_pct']:.1f}%")


def analyze_stuck_processes(days: int = 30):
    """Detect stuck/long-running processes with instance details"""
    print(f"\n{'='*80}")
    print("STUCK / LONG-RUNNING PROCESS ANALYSIS")
    print(f"{'='*80}\n")

    query = f"""
        WITH running AS (
            SELECT pi.id_ as instance_id, 
                   pi.proc_def_key_,
                   pi.business_key_,
                   EXTRACT(EPOCH FROM (NOW() - pi.start_time_)) as current_duration_seconds
            FROM act_hi_procinst pi
            WHERE pi.end_time_ IS NULL
        ), base AS (
            SELECT proc_def_key_,
                   AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as avg_dur,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as p95_dur
            FROM act_hi_procinst
            WHERE end_time_ IS NOT NULL
              AND start_time_ > NOW() - INTERVAL '{days*3} days'
            GROUP BY proc_def_key_
            HAVING COUNT(*) >= 5
        )
        SELECT r.instance_id, r.proc_def_key_, r.business_key_, r.current_duration_seconds, 
               b.avg_dur, b.p95_dur
        FROM running r
        LEFT JOIN base b ON r.proc_def_key_ = b.proc_def_key_
        ORDER BY r.current_duration_seconds DESC
        LIMIT 1000
    """

    results = execute_query(query)
    if not results:
        print("No running instances")
        return []

    df = pd.DataFrame(results)
    for col in ['current_duration_seconds', 'avg_dur', 'p95_dur']:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    stuck_list = []
    status_counts = {'critical': 0, 'warning': 0, 'attention': 0, 'normal': 0}

    for _, row in df.iterrows():
        cur = row['current_duration_seconds'] or 0
        p95 = row.get('p95_dur')

        if p95 and p95 > 0:
            if cur > p95 * 3:
                status = 'critical'
            elif cur > p95 * 2:
                status = 'warning'
            elif cur > (row.get('avg_dur') or p95) * 2:
                status = 'attention'
            else:
                status = 'normal'
        else:
            status = 'normal'

        status_counts[status] += 1

        if status in ['critical', 'warning']:
            business_key = row.get('business_key_', 'N/A')
            stuck_list.append({
                'instance_id': row['instance_id'],
                'process_key': row['proc_def_key_'],
                'business_key': business_key,
                'current_hours': cur / 3600,
                'p95_hours': p95 / 3600 if p95 else None,
                'status': status
            })

    print(f"Status distribution:")
    for st, cnt in status_counts.items():
        if cnt > 0:
            print(f"  {st}: {cnt}")

    if stuck_list:
        print(f"\n⚠️  Critical/Warning instances (top 15):")
        for s in stuck_list[:15]:
            bkey_str = f" | BKey: {s['business_key']}" if s['business_key'] != 'N/A' else ""
            print(f"  [{s['instance_id']}] {s['process_key']}: {s['current_hours']:.2f}h (P95: {s['p95_hours']:.2f}h) - {s['status']}{bkey_str}")

    return stuck_list


def analyze_activity_bottlenecks(days: int = 30):
    """Analyze slowest activities within processes"""
    print(f"\n{'='*80}")
    print("ACTIVITY BOTTLENECK ANALYSIS")
    print(f"{'='*80}\n")

    query = f"""
        SELECT 
            ai.proc_def_key_,
            ai.act_id_,
            ai.act_name_,
            ai.act_type_,
            COUNT(*) as execution_count,
            AVG(EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as avg_duration_s,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as p50_duration_s,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as p95_duration_s,
            STDDEV(EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as stddev_duration_s,
            MAX(EXTRACT(EPOCH FROM (ai.end_time_ - ai.start_time_))) as max_duration_s
        FROM act_hi_actinst ai
        WHERE ai.start_time_ > NOW() - INTERVAL '{days} days'
          AND ai.end_time_ IS NOT NULL
          AND ai.act_type_ IN ('serviceTask', 'sendTask', 'businessRuleTask', 'scriptTask', 'userTask')
        GROUP BY ai.proc_def_key_, ai.act_id_, ai.act_name_, ai.act_type_
        HAVING COUNT(*) >= 5
        ORDER BY avg_duration_s DESC
        LIMIT 30
    """

    results = execute_query(query)
    if not results:
        print("No activity data found")
        return []

    activities = []
    print("TOP 15 SLOWEST ACTIVITIES:\n")

    for idx, row in enumerate(results[:15], 1):
        avg_s = safe_float(row['avg_duration_s']) or 0
        p50_s = safe_float(row['p50_duration_s']) or 0
        p95_s = safe_float(row['p95_duration_s']) or 0
        stddev_s = safe_float(row['stddev_duration_s']) or 0
        max_s = safe_float(row['max_duration_s']) or 0
        count = int(row['execution_count'])

        cv = (stddev_s / avg_s) if avg_s > 0 else 0
        stability = "Stable" if cv < 0.3 else "Moderate" if cv < 1.0 else "Variable"

        activities.append({
            'process_key': row['proc_def_key_'],
            'activity_id': row['act_id_'],
            'activity_name': row['act_name_'],
            'activity_type': row['act_type_'],
            'execution_count': count,
            'avg_duration_s': avg_s,
            'p95_duration_s': p95_s,
            'cv': cv,
            'stability': stability
        })

        print(f"{idx}. {row['proc_def_key_']}::{row['act_name_'] or row['act_id_']} ({row['act_type_']})")
        print(f"   Executions: {count} | Avg: {avg_s:.2f}s | P95: {p95_s:.2f}s | Max: {max_s:.2f}s")
        print(f"   Variability: {stability} (CV: {cv:.3f})")
        print()

    return activities


def analyze_version_performance(days: int = 90):
    """Analyze performance changes between process versions"""
    print(f"\n{'='*80}")
    print("VERSION PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")

    query = f"""
        SELECT 
            pi.proc_def_key_,
            SUBSTRING(pi.proc_def_id_ FROM ':([0-9]+):') as version,
            COUNT(*) as instance_count,
            AVG(EXTRACT(EPOCH FROM (pi.end_time_ - pi.start_time_))) as avg_duration_s,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (pi.end_time_ - pi.start_time_))) as p95_duration_s,
            STDDEV(EXTRACT(EPOCH FROM (pi.end_time_ - pi.start_time_))) as stddev_s
        FROM act_hi_procinst pi
        WHERE pi.start_time_ > NOW() - INTERVAL '{days} days'
          AND pi.end_time_ IS NOT NULL
          AND pi.end_time_ > pi.start_time_
        GROUP BY pi.proc_def_key_, version
        HAVING COUNT(*) >= 5
        ORDER BY pi.proc_def_key_, version DESC
    """

    results = execute_query(query)
    if not results:
        print("No version data found")
        return []

    version_comparisons = []

    # Group by process
    by_process = {}
    for row in results:
        proc_key = row['proc_def_key_']
        if proc_key not in by_process:
            by_process[proc_key] = []
        by_process[proc_key].append(row)

    print("PROCESSES WITH VERSION CHANGES:\n")

    for proc_key, versions in by_process.items():
        if len(versions) < 2:
            continue

        # Sort by version
        versions = sorted(versions, key=lambda x: x['version'], reverse=True)
        latest = versions[0]
        previous = versions[1] if len(versions) > 1 else None

        latest_avg = safe_float(latest['avg_duration_s']) or 0
        previous_avg = safe_float(previous['avg_duration_s']) or 0 if previous else 0

        if previous_avg > 0:
            change_pct = ((latest_avg - previous_avg) / previous_avg) * 100
            direction = "📈 SLOWER" if change_pct > 10 else "📉 FASTER" if change_pct < -10 else "→ STABLE"

            version_comparisons.append({
                'process_key': proc_key,
                'latest_version': latest['version'],
                'previous_version': previous['version'],
                'change_pct': change_pct,
                'latest_avg_s': latest_avg,
                'previous_avg_s': previous_avg
            })

            if abs(change_pct) > 10:  # Only print significant changes
                print(f"{proc_key}")
                print(f"  Version {latest['version']}: {latest_avg:.2f}s (n={int(latest['instance_count'])})")
                print(f"  Version {previous['version']}: {previous_avg:.2f}s (n={int(previous['instance_count'])})")
                print(f"  Change: {direction} ({change_pct:+.1f}%)")
                print()

    if not version_comparisons:
        print("No significant version changes detected")

    return version_comparisons


def analyze_outlier_patterns(days: int = 90):
    """Analyze outlier patterns for anomaly detection (with internal parallelization)"""
    print(f"\n{'='*80}")
    print("OUTLIER & ANOMALY PATTERNS")
    print(f"{'='*80}\n")

    query = f"""
        SELECT 
            proc_def_key_,
            COUNT(*) as total_count,
            AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as avg_s,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as q3_s,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as q1_s
        FROM act_hi_procinst
        WHERE start_time_ > NOW() - INTERVAL '{days} days'
          AND end_time_ IS NOT NULL
          AND end_time_ > start_time_
        GROUP BY proc_def_key_
        HAVING COUNT(*) >= 10
    """

    results = execute_query(query)
    if not results:
        print("No data for outlier analysis")
        return []

    outlier_analysis = []
    print("OUTLIER DETECTION THRESHOLDS BY PROCESS:\n")

    # Function to analyze single process outliers
    def analyze_single_process_outliers(row):
        q1 = safe_float(row['q1_s']) or 0
        q3 = safe_float(row['q3_s']) or 0
        avg = safe_float(row['avg_s']) or 0
        total = int(row['total_count'])

        iqr = q3 - q1

        # Standard IQR outlier detection
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Extreme outlier detection (3 * IQR)
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr

        # Count actual outliers
        query_outliers = f"""
            SELECT 
                COUNT(*) as outlier_count,
                COUNT(CASE WHEN d > {extreme_upper} THEN 1 END) as extreme_count
            FROM (
                SELECT EXTRACT(EPOCH FROM (end_time_ - start_time_)) as d
                FROM act_hi_procinst
                WHERE proc_def_key_ = '{row['proc_def_key_']}'
                  AND start_time_ > NOW() - INTERVAL '{days} days'
                  AND end_time_ IS NOT NULL
            ) sub
            WHERE d > {upper_bound}
        """

        outlier_results = execute_query(query_outliers)
        outlier_count = int(outlier_results[0]['outlier_count']) if outlier_results else 0
        extreme_count = int(outlier_results[0]['extreme_count']) if outlier_results else 0
        outlier_pct = (outlier_count / total * 100) if total > 0 else 0

        return {
            'process_key': row['proc_def_key_'],
            'total_count': total,
            'outlier_count': outlier_count,
            'outlier_pct': outlier_pct,
            'extreme_outlier_count': extreme_count,
            'iqr_upper_threshold_s': upper_bound,
            'extreme_upper_threshold_s': extreme_upper,
            'mean_duration_s': avg,
            'lower_bound': lower_bound
        }

    # Process in parallel (internal to this function)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_process = {executor.submit(analyze_single_process_outliers, row): row for row in results}

        for future in concurrent.futures.as_completed(future_to_process):
            result = future.result()
            outlier_analysis.append(result)

    # Sort by process key for consistent output
    outlier_analysis.sort(key=lambda x: x['process_key'])

    # Print results
    for analysis in outlier_analysis:
        if analysis['outlier_pct'] > 5 or analysis['extreme_outlier_count'] > 0:
            print(f"{analysis['process_key']}")
            print(f"  Total instances: {analysis['total_count']}")
            print(f"  IQR thresholds: [{max(0.0, analysis['lower_bound']):.2f}s, {analysis['iqr_upper_threshold_s']:.2f}s]")
            print(f"  Outliers: {analysis['outlier_count']} ({analysis['outlier_pct']:.1f}%)")
            if analysis['extreme_outlier_count'] > 0:
                print(f"  ⚠️  Extreme outliers: {analysis['extreme_outlier_count']} (beyond {analysis['extreme_upper_threshold_s']:.2f}s)")
            print()

    return outlier_analysis


def analyze_incidents_and_errors(days: int = 30):
    """Analyze incidents, fallback to job errors"""
    print(f"\n{'='*80}")
    print("INCIDENT / ERROR ANALYSIS")
    print(f"{'='*80}\n")

    query = f"""
        SELECT pi.proc_def_key_, pi.id_ as instance_id, pi.business_key_,
               inc.incident_type_, inc.incident_msg_,
               CASE WHEN inc.end_time_ IS NULL THEN 'open' ELSE 'resolved' END as status,
               EXTRACT(EPOCH FROM (COALESCE(inc.end_time_, NOW()) - inc.create_time_)) as duration_seconds
        FROM act_hi_incident inc
        JOIN act_hi_procinst pi ON inc.proc_inst_id_ = pi.id_
        WHERE inc.create_time_ > NOW() - INTERVAL '{days} days'
        LIMIT 5000
    """

    results = execute_query(query)
    source = "incidents"

    if not results:
        print("No incidents found, checking job errors...")

        # Fixed: Use correct job_log columns (no end_time_ on job_log)
        query = f"""
            SELECT 
                jl.process_def_key_ as proc_def_key_,
                jl.process_instance_id_ as instance_id,
                pi.business_key_,
                COALESCE(jl.job_def_type_, 'unknown') as incident_type_,
                jl.job_exception_msg_ as incident_msg_,
                CASE 
                    WHEN jl.job_exception_msg_ IS NOT NULL THEN 'open'
                    ELSE 'resolved'
                END as status,
                0 as duration_seconds
            FROM act_hi_job_log jl
            LEFT JOIN act_hi_procinst pi ON jl.process_instance_id_ = pi.id_
            WHERE jl.timestamp_ > NOW() - INTERVAL '{days} days'
              AND jl.job_exception_msg_ IS NOT NULL
            LIMIT 5000
        """

        results = execute_query(query)
        source = "job_errors"

        if not results:
            print("No job errors found, analyzing activity failures...")

            query = f"""
                SELECT 
                    ai.proc_def_key_,
                    ai.id_ as instance_id,
                    pi.business_key_,
                    ai.act_type_ as incident_type_,
                    COUNT(*) as failure_count
                FROM act_hi_actinst ai
                JOIN act_hi_procinst pi ON ai.proc_inst_id_ = pi.id_
                WHERE ai.start_time_ > NOW() - INTERVAL '{days} days'
                  AND ai.act_type_ IN ('serviceTask', 'sendTask', 'businessRuleTask', 'scriptTask', 'userTask')
                GROUP BY ai.proc_def_key_, ai.id_, pi.business_key_, ai.act_type_
                HAVING COUNT(*) >= 5
                ORDER BY failure_count DESC
            """

            results = execute_query(query)
            source = "activity_failures"

            if not results:
                print("No error data found")
                return {}

    print(f"Found {len(results)} {source} to analyze\n")

    process_errors = {}
    total_open = 0
    total_resolved = 0

    for row in results:
        proc_key = row.get('proc_def_key_', 'unknown')
        status = row.get('status', 'unknown')
        instance_id = row.get('instance_id', 'N/A')
        business_key = row.get('business_key_', 'N/A')

        if proc_key not in process_errors:
            process_errors[proc_key] = {
                'open': 0,
                'resolved': 0,
                'error_types': set(),
                'total_duration_s': 0,
                'count': 0,
                'sample_instances': []
            }

        if status == 'open':
            process_errors[proc_key]['open'] += 1
            total_open += 1
        else:
            process_errors[proc_key]['resolved'] += 1
            total_resolved += 1

        # Track error types
        if row.get('incident_type_'):
            process_errors[proc_key]['error_types'].add(str(row['incident_type_']))

        # Track duration
        if row.get('duration_seconds'):
            process_errors[proc_key]['total_duration_s'] += float(row['duration_seconds'])

        # Store sample instances (keep first 3)
        if len(process_errors[proc_key]['sample_instances']) < 3:
            process_errors[proc_key]['sample_instances'].append({
                'instance_id': instance_id,
                'business_key': business_key
            })

        process_errors[proc_key]['count'] += 1

    print(f"Total: {total_open + total_resolved} errors")
    print(f"  Open: {total_open}")
    print(f"  Resolved: {total_resolved}")

    if process_errors:
        print(f"\nTop processes with errors:")
        sorted_procs = sorted(process_errors.items(),
                             key=lambda x: x[1]['open'] + x[1]['resolved'],
                             reverse=True)[:10]
        for proc_key, data in sorted_procs:
            total = data['open'] + data['resolved']
            avg_duration = data['total_duration_s'] / data['count'] / 3600 if data['count'] > 0 else 0
            error_types = ', '.join(list(data['error_types'])[:2])
            print(f"  {proc_key}: {total} errors ({data['open']} open, {data['resolved']} resolved)")
            print(f"    Avg error duration: {avg_duration:.2f}h | Types: {error_types}")

            # Show sample instances
            if data['sample_instances']:
                print(f"    Sample instances:")
                for inst in data['sample_instances'][:2]:
                    bkey_str = f" | BKey: {inst['business_key']}" if inst['business_key'] != 'N/A' else ""
                    print(f"      - ID: {inst['instance_id']}{bkey_str}")

    return process_errors


# ============================================================================
# ENTERPRISE CAMUNDA LOAD PATTERN & VERSION ANALYSIS
# ============================================================================

def analyze_load_patterns(days: int = 90):
    """Analyze enterprise Camunda 7 load patterns: business days, working hours, peak times"""
    print(f"\n{'='*80}")
    print("ENTERPRISE CAMUNDA LOAD PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    # Daily patterns (business vs weekend)
    day_query = f"""
        SELECT 
            EXTRACT(DOW FROM start_time_) as day_of_week,
            CASE EXTRACT(DOW FROM start_time_)
                WHEN 0 THEN 'Sunday (Weekend)'
                WHEN 1 THEN 'Monday (Business)'
                WHEN 2 THEN 'Tuesday (Business)'
                WHEN 3 THEN 'Wednesday (Business)'
                WHEN 4 THEN 'Thursday (Business)'
                WHEN 5 THEN 'Friday (Business)'
                WHEN 6 THEN 'Saturday (Weekend)'
            END as day_name,
            COUNT(*) as total_instances,
            AVG(EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_))) as avg_duration_s,
            MAX(EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_))) as max_duration_s
        FROM act_hi_procinst
        WHERE start_time_ > NOW() - INTERVAL '{days} days'
        GROUP BY EXTRACT(DOW FROM start_time_)
        ORDER BY day_of_week
    """

    day_results = execute_query(day_query)

    if not day_results:
        print("No load pattern data found")
        return {}

    print("DAILY PATTERNS (Business Days vs Weekends):\n")

    business_totals = {'instances': 0, 'duration': 0, 'count': 0}
    weekend_totals = {'instances': 0, 'duration': 0, 'count': 0}

    for row in day_results:
        day_name = row.get('day_name', '')
        total = int(row['total_instances'])
        avg_dur = float(row['avg_duration_s']) if row.get('avg_duration_s') else 0
        max_dur = float(row['max_duration_s']) if row.get('max_duration_s') else 0

        is_weekend = 'Weekend' in day_name

        if is_weekend:
            weekend_totals['instances'] += total
            weekend_totals['duration'] += avg_dur
            weekend_totals['count'] += 1
        else:
            business_totals['instances'] += total
            business_totals['duration'] += avg_dur
            business_totals['count'] += 1

        pattern = "🟡 Weekend" if is_weekend else "🔵 Business"
        print(f"  {pattern:18} {day_name:20} {total:,} instances | Avg: {avg_dur:.2f}s | Max: {max_dur:.2f}s")

    # Calculate patterns
    business_avg = 0
    if business_totals['count'] > 0:
        business_avg = business_totals['duration'] / business_totals['count']
        print(f"\n📊 Business Days Average: {business_avg:.2f}s/instance | Total: {business_totals['instances']:,} instances")

    weekend_avg = 0
    if weekend_totals['count'] > 0:
        weekend_avg = weekend_totals['duration'] / weekend_totals['count']
        print(f"📊 Weekends Average: {weekend_avg:.2f}s/instance | Total: {weekend_totals['instances']:,} instances")

        if business_totals['count'] > 0 and business_avg > 0:
            factor = weekend_avg / business_avg
            diff_pct = (factor - 1) * 100
            direction = "SLOWER ⬆️" if factor > 1.1 else "FASTER ⬇️" if factor < 0.9 else "SIMILAR"
            print(f"  → Weekend vs Business: {direction} ({diff_pct:+.1f}%)")

    # Hourly patterns
    hour_query = f"""
        SELECT 
            EXTRACT(HOUR FROM start_time_) as hour_of_day,
            COUNT(*) as hourly_instances,
            AVG(EXTRACT(EPOCH FROM (COALESCE(end_time_, NOW()) - start_time_))) as avg_duration_s
        FROM act_hi_procinst
        WHERE start_time_ > NOW() - INTERVAL '{days} days'
        GROUP BY EXTRACT(HOUR FROM start_time_)
        ORDER BY hourly_instances DESC
    """

    hour_results = execute_query(hour_query)

    if hour_results:
        print(f"\n⏰ PEAK LOAD TIMES (Top 10 by volume):\n")

        for idx, row in enumerate(hour_results[:10], 1):
            hour = int(row['hour_of_day'])
            count = int(row['hourly_instances'])
            avg = float(row['avg_duration_s']) if row.get('avg_duration_s') else 0

            # Business hours classification
            is_business_hours = 7 <= hour <= 18
            period = "🏢 Business" if is_business_hours else "🌙 After-hrs"

            print(f"  {idx:2}. {period} {hour:02d}:00-{hour+1:02d}:00 - {count:,} instances | Avg: {avg:.2f}s")

    return {
        'business_instances': business_totals['instances'],
        'weekend_instances': weekend_totals['instances'],
        'peak_hour': hour_results[0]['hour_of_day'] if hour_results else 0,
        'peak_hour_count': int(hour_results[0]['hourly_instances']) if hour_results else 0
    }


def analyze_version_all(days: int = 180):
    """Analyze ALL process versions in the period (not just latest 2)"""
    print(f"\n{'='*80}")
    print("COMPLETE VERSION PERFORMANCE ANALYSIS (ALL VERSIONS)")
    print(f"{'='*80}\n")

    query = f"""
        SELECT 
            pi.proc_def_key_,
            SUBSTRING(pi.proc_def_id_ FROM ':([0-9]+):') as version,
            COUNT(*) as instance_count,
            AVG(EXTRACT(EPOCH FROM (end_time_ - start_time_))) as avg_duration_s,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (end_time_ - start_time_))) as p95_duration_s,
            MIN(start_time_) as first_used,
            MAX(start_time_) as last_used
        FROM act_hi_procinst pi
        WHERE pi.start_time_ > NOW() - INTERVAL '{days} days'
          AND pi.end_time_ IS NOT NULL
          AND pi.end_time_ > pi.start_time_
        GROUP BY pi.proc_def_key_, SUBSTRING(pi.proc_def_id_ FROM ':([0-9]+):')
        ORDER BY pi.proc_def_key_, CAST(SUBSTRING(pi.proc_def_id_ FROM ':([0-9]+):') AS INTEGER) DESC
    """

    results = execute_query(query)
    if not results:
        print("No version data found")
        return []

    # Group by process
    by_process = {}
    for row in results:
        proc_key = row['proc_def_key_']
        if proc_key not in by_process:
            by_process[proc_key] = []
        by_process[proc_key].append(row)

    multi_version_count = 0
    version_analysis = []

    print("PROCESSES WITH MULTIPLE VERSIONS (Complete History):\n")

    for proc_key in sorted(by_process.keys()):
        versions = by_process[proc_key]

        if len(versions) < 2:
            continue

        multi_version_count += 1

        # Sort by version number descending
        try:
            versions_sorted = sorted(versions, key=lambda x: int(x['version']), reverse=True)
        except:
            versions_sorted = versions

        print(f"{proc_key} - {len(versions)} versions:")

        for idx, row in enumerate(versions_sorted):
            v = row['version']
            count = int(row['instance_count'])
            avg_s = float(row['avg_duration_s']) or 0
            p95_s = float(row['p95_duration_s']) or 0
            first_used = row.get('first_used')
            last_used = row.get('last_used')

            # Compare with previous version if exists
            if idx > 0 and idx < len(versions_sorted):
                prev_avg = float(versions_sorted[idx-1]['avg_duration_s']) or 0
                if prev_avg > 0:
                    change_pct = ((avg_s - prev_avg) / prev_avg) * 100
                    if abs(change_pct) > 15:
                        direction = "📈 SLOWER" if change_pct > 0 else "📉 FASTER"
                        print(f"  v{v}: {avg_s:.2f}s (P95: {p95_s:.2f}s) | {count:,} inst | {direction} ({change_pct:+.1f}%)")
                    else:
                        print(f"  v{v}: {avg_s:.2f}s (P95: {p95_s:.2f}s) | {count:,} inst | → STABLE")
                else:
                    print(f"  v{v}: {avg_s:.2f}s (P95: {p95_s:.2f}s) | {count:,} inst")
            else:
                marker = " (Current)" if idx == 0 else " (Previous)"
                print(f"  v{v}: {avg_s:.2f}s (P95: {p95_s:.2f}s) | {count:,} inst{marker}")

            version_analysis.append({
                'process_key': proc_key,
                'version': v,
                'avg_duration_s': avg_s,
                'p95_duration_s': p95_s,
                'instance_count': count,
                'first_used': first_used,
                'last_used': last_used
            })

        print()

    print(f"Total processes with multiple versions: {multi_version_count}\n")

    return version_analysis


# ============================================================================
# INITIALIZATION & MAIN
# ============================================================================

def initialize_database():
    """Initialize database"""
    load_dotenv()
    db_config = {
        'dbname': os.getenv('DB_NAME', 'camunda'),
        'user': os.getenv('DB_USER', 'camunda'),
        'password': os.getenv('DB_PASSWORD', 'camunda'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
    }

    print("Initializing database connection...")
    init_db_pool(db_config=db_config)
    print("✓ Database initialized\n")


def main():
    """Main execution"""
    initialize_database()

    print(f"\n{'='*80}")
    print("CAMUNDA PROFESSIONAL ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print("\nFocus: Business-critical processes only (excluding ultra_fast internal processes)")
    print(f"Analysis period: 180 days\n")

    # Phase 1: Quick categorization
    process_categories = categorize_all_processes(days=180)

    if not process_categories:
        print("No processes found to analyze")
        return

    # Phase 2: Detailed analysis
    per_process_summary = analyze_business_critical_processes(process_categories, days=180)

    # Phase 3: Print analysis
    if per_process_summary:
        print_detailed_analysis(per_process_summary)

    # Phase 4: Extended analyses (sequential for clean output)
    print(f"\n{'='*80}\nEXTENDED ANALYSIS PHASE")
    print(f"{'='*80}\n")

    # Activity bottleneck analysis
    activities = analyze_activity_bottlenecks(days=30)

    # Version performance comparison
    versions = analyze_version_performance(days=90)

    # Outlier detection (uses internal parallelization)
    outliers = analyze_outlier_patterns(days=90)

    # Stuck process detection
    stuck_summary = analyze_stuck_processes(days=30)

    # Incident/error analysis with fallback
    incident_summary = analyze_incidents_and_errors(days=30)

    # Enterprise load pattern analysis
    load_patterns = analyze_load_patterns(days=90)

    # Complete version analysis
    complete_version_analysis = analyze_version_all(days=180)

    # Final summary and recommendations
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE - PROFESSIONAL SUMMARY")
    print(f"{'='*80}\n")

    if per_process_summary:
        total = len(per_process_summary)
        by_cat = {}
        for p in per_process_summary:
            cat = p['category']
            by_cat[cat] = by_cat.get(cat, 0) + 1

        print(f"✓ Business-critical processes analyzed: {total}")
        for cat in ANALYZE_CATEGORIES:
            count = by_cat.get(cat, 0)
            if count > 0:
                print(f"  {CATEGORY_LABELS[cat]}: {count}")

    print(f"\n✓ Extended Analysis Results:")
    print(f"  Activity bottlenecks identified: {len(activities)}")
    print(f"  Version performance changes: {len([v for v in versions if abs(v.get('change_pct', 0)) > 20])}")
    print(f"  Processes with outliers: {len([o for o in outliers if o['outlier_pct'] > 5])}")
    print(f"  Stuck instances: {len(stuck_summary) if stuck_summary else 0}")
    print(f"  Error sources: {len(incident_summary) if incident_summary else 0}")

    if load_patterns:
        print(f"  Load pattern analysis: Business={load_patterns.get('business_instances', 0):,}, Weekend={load_patterns.get('weekend_instances', 0):,}, Peak Hour={load_patterns.get('peak_hour', 'N/A')}")

    if complete_version_analysis:
        print(f"  Complete version history: {len(complete_version_analysis)} version records analyzed")

    print(f"\n✓ RECOMMENDATIONS FOR AI/ML CONFIGURATION:")
    print(f"  1. Use category-specific thresholds (see CATEGORY_LABELS above)")
    print(f"  2. Apply IQR-based outlier filtering in predictions")
    print(f"  3. Track version performance regressions (>20% change)")
    print(f"  4. Monitor activity bottlenecks for optimization opportunities")
    print(f"  5. Use P95 for fast processes, P99 for batch processes")
    print(f"  6. Implement dynamic stuck detection per category")
    print(f"  7. Factor error patterns into failure prediction models")
    print(f"  8. Apply load-aware SLA adjustments (business vs weekend)")
    print(f"  9. Schedule batch jobs during low-load hours (see peak times)")

    # Critical insights section
    generate_critical_insights(per_process_summary, activities, versions, outliers, stuck_summary, incident_summary, load_patterns)

    print(f"\n✓ Analysis complete. Use data above to configure AI/ML in ai_service.py")

    # Cleanup
    try:
        get_db_helper().close()
        print("✓ Database connection closed")
    except Exception:
        pass


def generate_critical_insights(per_process_summary, activities, versions, outliers, stuck_summary, incident_summary, load_patterns):
    """Generate critical insights and specific recommendations based on analysis"""
    print(f"\n{'='*80}")
    print("🎯 CRITICAL INSIGHTS & SPECIFIC RECOMMENDATIONS")
    print(f"{'='*80}\n")

    insights = []

    # 1. Identify processes with extreme P95 vs median gaps
    print("1️⃣  PROCESSES WITH EXTREME VARIABILITY (P95 >> Median):\n")
    extreme_variance = []
    for p in per_process_summary:
        stats = p['stats']
        median_s = stats['median_s']
        p95_s = stats['p95_s']

        if median_s > 0:
            variance_ratio = p95_s / median_s
            if variance_ratio > 100:  # P95 is 100x+ median
                extreme_variance.append({
                    'process': p['process_key'],
                    'median': median_s,
                    'p95': p95_s,
                    'ratio': variance_ratio,
                    'category': p['category']
                })

    extreme_variance.sort(key=lambda x: x['ratio'], reverse=True)

    for ev in extreme_variance[:5]:
        print(f"  ⚠️  {ev['process']}")
        print(f"     Median: {ev['median']:.2f}s | P95: {ev['p95']:.2f}s | Ratio: {ev['ratio']:.0f}x")
        print(f"     💡 Action: Investigate why some instances take {ev['ratio']:.0f}x longer")
        print(f"        - Check for external dependencies causing delays")
        print(f"        - Look for error retries or stuck subprocess")
        print(f"        - Consider splitting into fast-path and slow-path variants")
        print()

    # 2. Version regressions requiring immediate attention
    print("\n2️⃣  VERSION REGRESSIONS REQUIRING IMMEDIATE ACTION:\n")
    critical_regressions = []
    for v in versions:
        if v.get('change_pct', 0) > 100:  # More than 100% slower
            critical_regressions.append(v)

    if critical_regressions:
        critical_regressions.sort(key=lambda x: x['change_pct'], reverse=True)
        for reg in critical_regressions[:3]:
            print(f"  🚨 {reg['process_key']}")
            print(f"     Version {reg['previous_version']}: {reg['previous_avg_s']:.2f}s")
            print(f"     Version {reg['latest_version']}: {reg['latest_avg_s']:.2f}s")
            print(f"     Degradation: {reg['change_pct']:+.1f}%")
            print(f"     💡 Action: ROLLBACK to version {reg['previous_version']} immediately")
            print(f"        - Investigate code changes between versions")
            print(f"        - Check database query performance")
            print(f"        - Review external API changes")
            print()
    else:
        print("  ✅ No critical regressions detected\n")

    # 3. Stuck instances analysis
    print("\n3️⃣  STUCK INSTANCE DEEP DIVE:\n")
    if stuck_summary and len(stuck_summary) > 0:
        # Group by process
        by_process = {}
        for s in stuck_summary:
            proc = s['process_key']
            if proc not in by_process:
                by_process[proc] = []
            by_process[proc].append(s)

        print(f"  Total critical/warning stuck instances: {len(stuck_summary)}")
        print(f"  Affected processes: {len(by_process)}")
        print()

        # Find processes with most stuck instances
        process_stuck_count = [(k, len(v)) for k, v in by_process.items()]
        process_stuck_count.sort(key=lambda x: x[1], reverse=True)

        for proc, count in process_stuck_count[:3]:
            oldest = max(by_process[proc], key=lambda x: x['current_hours'])
            print(f"  🔴 {proc}: {count} stuck instances")
            print(f"     Oldest stuck: {oldest['current_hours']:.0f}h ({oldest['current_hours']/24:.0f} days)")
            print(f"     Business Key: {oldest['business_key']}")
            print(f"     Instance ID: {oldest['instance_id']}")
            print(f"     💡 Action: Investigate this specific instance first")
            print(f"        - Check for waiting external callbacks")
            print(f"        - Look for missing message correlation")
            print(f"        - Review stuck user tasks or timers")
            print()

    # 4. Load pattern insights
    print("\n4️⃣  LOAD PATTERN OPTIMIZATION OPPORTUNITIES:\n")
    if load_patterns:
        business_avg = load_patterns.get('business_instances', 0)
        weekend_avg = load_patterns.get('weekend_instances', 0)
        peak_hour = load_patterns.get('peak_hour', 0)

        if weekend_avg > 0 and business_avg > 0:
            weekend_ratio = weekend_avg / (business_avg + weekend_avg) * 100
            print(f"  📊 Weekend load: {weekend_ratio:.1f}% of total traffic")

            if weekend_ratio < 20:
                print(f"  💡 Action: Schedule batch jobs on weekends")
                print(f"     - Data cleanup processes")
                print(f"     - Report generation")
                print(f"     - Archive operations")
                print()

        if peak_hour:
            print(f"  ⏰ Peak hour: {int(peak_hour):02d}:00")
            print(f"  💡 Action: Optimize for peak load")
            print(f"     - Pre-warm caches before {int(peak_hour):02d}:00")
            print(f"     - Scale up resources at {int(peak_hour)-1:02d}:00")
            print(f"     - Delay non-critical jobs until after {int(peak_hour)+2:02d}:00")
            print()

    # 5. Outlier-heavy processes
    print("\n5️⃣  PROCESSES WITH EXCESSIVE OUTLIERS (>15%):\n")
    high_outliers = [o for o in outliers if o['outlier_pct'] > 15]
    high_outliers.sort(key=lambda x: x['outlier_pct'], reverse=True)

    for o in high_outliers[:5]:
        print(f"  ⚠️  {o['process_key']}: {o['outlier_pct']:.1f}% outliers")
        print(f"     Total: {o['total_count']:,} | Outliers: {o['outlier_count']:,}")
        print(f"     Normal threshold: {o['iqr_upper_threshold_s']:.2f}s")
        print(f"     💡 Action: This process is HIGHLY unpredictable")
        print(f"        - Don't use for SLA-critical workflows")
        print(f"        - Implement timeout safeguards")
        print(f"        - Add monitoring alerts for P99 violations")
        print()

    # 6. Activity bottleneck focus
    print("\n6️⃣  TOP 3 OPTIMIZATION TARGETS (Activity Level):\n")
    if activities:
        # Focus on high-execution, high-duration activities
        optimization_targets = []
        for act in activities:
            impact_score = act['execution_count'] * act['avg_duration_s']
            optimization_targets.append({
                **act,
                'impact_score': impact_score
            })

        optimization_targets.sort(key=lambda x: x['impact_score'], reverse=True)

        for idx, target in enumerate(optimization_targets[:3], 1):
            hours_wasted = target['execution_count'] * target['avg_duration_s'] / 3600
            print(f"  {idx}. {target['process_key']}::{target['activity_name']}")
            print(f"     Type: {target['activity_type']} | Executions: {target['execution_count']:,}")
            print(f"     Avg: {target['avg_duration_s']:.2f}s | P95: {target['p95_duration_s']:.2f}s")
            print(f"     Total time spent: {hours_wasted:.0f} hours")
            print(f"     💡 Action: 10% improvement saves {hours_wasted * 0.1:.0f} hours")

            if target['activity_type'] == 'userTask':
                print(f"        - Reduce manual work with automation")
                print(f"        - Add decision support tools")
                print(f"        - Implement auto-assignment rules")
            elif target['activity_type'] == 'serviceTask':
                print(f"        - Optimize external API calls")
                print(f"        - Add caching layer")
                print(f"        - Parallelize where possible")
            print()

    # 7. Weekend vs business pattern recommendation
    print("\n7️⃣  ENTERPRISE SCHEDULING STRATEGY:\n")
    business_avg = load_patterns.get('business_instances', 0) if load_patterns else 0
    weekend_avg = load_patterns.get('weekend_instances', 0) if load_patterns else 0
    peak_hour = load_patterns.get('peak_hour', 0) if load_patterns else 0

    print(f"  📅 Business days handle {business_avg:,} instances")
    print(f"  📅 Weekends handle {weekend_avg:,} instances")
    print(f"  💡 Action: Implement load-aware scheduling")
    print(f"     - Business hours (07:00-19:00): Real-time priority")
    print(f"     - After-hours (19:00-07:00): Batch jobs ok")
    print(f"     - Weekends: Heavy batch processing window")
    if peak_hour:
        print(f"     - Peak hour ({int(peak_hour):02d}:00): Minimal new process starts")
    print()

    return insights


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            get_db_helper().close()
        except Exception:
            pass
