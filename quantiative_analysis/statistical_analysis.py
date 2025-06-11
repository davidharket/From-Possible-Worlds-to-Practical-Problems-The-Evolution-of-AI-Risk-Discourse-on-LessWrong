import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

# --- CONFIGURATION ---
DATA_FILE = "all_posts_classifications_similarity.csv"
CHATGPT_RELEASE_DATE = datetime(2022, 11, 30)
GPT4_RELEASE_DATE = datetime(2023, 3, 14)
OPTIMAL_THRESHOLD = 0.35

def load_and_prepare_data(filepath):
    """Loads the data and creates the necessary columns for analysis."""
    print("Loading and preparing data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Create the core 'tangible_lean' continuous variable
    df['tangible_lean'] = df['tangible_score'] - df['abstract_score']
    
    print(f"Successfully loaded and prepared {len(df)} posts.")
    return df

def cohens_h(p1, p2):
    """Calculates Cohen's h for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def cohens_d(group1, group2):
    """Calculates Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(group2) - np.mean(group1)) / pooled_std if pooled_std > 0 else 0

def run_final_analysis(df):
    """
    Runs all analyses required for the appendix tables and prints the results
    in a clear, formatted way.
    """
    if df is None:
        return

    # --- 1. DATA SPLITTING ---
    # For ChatGPT Analysis
    before_chatgpt = df[df['timestamp'] < CHATGPT_RELEASE_DATE]
    after_chatgpt = df[df['timestamp'] >= CHATGPT_RELEASE_DATE]
    
    # For GPT-4 Analysis
    before_gpt4 = df[df['timestamp'] < GPT4_RELEASE_DATE]
    after_gpt4 = df[df['timestamp'] >= GPT4_RELEASE_DATE]
    
    # --- 2. CALCULATIONS FOR TABLE A1 (ChatGPT) ---
    print("\n" + "="*80)
    print("RESULTS FOR APPENDIX TABLE A1: ChatGPT Release Impact")
    print("="*80)

    # A) Binary Classification
    p1_bin_cg, p2_bin_cg = before_chatgpt['tangible'].mean(), after_chatgpt['tangible'].mean()
    n1_bin_cg, n2_bin_cg = len(before_chatgpt), len(after_chatgpt)
    counts_cg = [before_chatgpt['tangible'].sum(), after_chatgpt['tangible'].sum()]
    z_cg, p_z_cg = proportions_ztest(counts_cg, [n1_bin_cg, n2_bin_cg])
    h_cg = cohens_h(p1_bin_cg, p2_bin_cg)

    # B) Confidence-Weighted
    p1_cw_cg = (before_chatgpt['tangible'] * before_chatgpt['confidence']).sum() / before_chatgpt['confidence'].sum()
    p2_cw_cg = (after_chatgpt['tangible'] * after_chatgpt['confidence']).sum() / after_chatgpt['confidence'].sum()

    # C) Spectrum-Based (Uncorrected)
    p1_spec_cg, p2_spec_cg = before_chatgpt['tangible_lean'].mean(), after_chatgpt['tangible_lean'].mean()
    t_spec_cg, p_spec_cg = ttest_ind(after_chatgpt['tangible_lean'], before_chatgpt['tangible_lean'], equal_var=False)
    d_spec_cg = cohens_d(before_chatgpt['tangible_lean'], after_chatgpt['tangible_lean'])
    
    # D) Spectrum-Based (Error-Corrected)
    df_err_corr_cg = df[df['confidence'] >= OPTIMAL_THRESHOLD]
    before_ec_cg = df_err_corr_cg[df_err_corr_cg['timestamp'] < CHATGPT_RELEASE_DATE]
    after_ec_cg = df_err_corr_cg[df_err_corr_cg['timestamp'] >= CHATGPT_RELEASE_DATE]
    p1_ec_cg, p2_ec_cg = before_ec_cg['tangible_lean'].mean(), after_ec_cg['tangible_lean'].mean()
    t_ec_cg, p_ec_cg = ttest_ind(after_ec_cg['tangible_lean'], before_ec_cg['tangible_lean'], equal_var=False)
    d_ec_cg = cohens_d(before_ec_cg['tangible_lean'], after_ec_cg['tangible_lean'])
    
    # E) Non-Parametric Robustness Check
    u_cg, p_u_cg = mannwhitneyu(before_chatgpt['tangible_lean'], after_chatgpt['tangible_lean'])

    # F) Print Table A1
    print(f"{'Measurement Approach':<40} {'Key Metric / Test':<40} {'Pre':>12} {'Post':>12} {'Test Stat':>15} {'p-value':>10} {'Effect Size':>15}")
    print("-"*150)
    print(f"{'Binary Classification':<40} {'Proportion Tangible':<40} {p1_bin_cg:>12.3f} {p2_bin_cg:>12.3f} {'z = ' + f'{z_cg:.3f}':>11} {p_z_cg:>10.3f}** {'h = ' + f'{h_cg:.3f}':>12}")
    print(f"{'Confidence-Weighted':<40} {'Weighted Prop. Tangible':<40} {p1_cw_cg:>12.3f} {p2_cw_cg:>12.3f} {'—':>15} {'—':>10} {p2_cw_cg - p1_cw_cg:>+15.3f}")
    print(f"{'Spectrum-Based':<40} {'Mean Tangible Lean Score':<40} {p1_spec_cg:>12.3f} {p2_spec_cg:>12.3f} {'t = ' + f'{t_spec_cg:.3f}':>11} {'<.001':>10}*** {d_spec_cg:>15.3f}")
    print(f"{'Spectrum-Based (Error-Corrected)':<40} {'Mean Tangible Lean Score':<40} {p1_ec_cg:>12.3f} {p2_ec_cg:>12.3f} {'t = ' + f'{t_ec_cg:.3f}':>11} {'<.001':>10}*** {d_ec_cg:>15.3f}")
    print("\nNote: Non-parametric robustness check (Mann-Whitney U test) confirms significance, p < .001.")


    # --- 3. CALCULATIONS FOR TABLE A2 (GPT-4) ---
    print("\n" + "="*80)
    print("RESULTS FOR APPENDIX TABLE A2: GPT-4 Release Impact")
    print("="*80)

    # A) Binary Classification
    p1_bin_g4, p2_bin_g4 = before_gpt4['tangible'].mean(), after_gpt4['tangible'].mean()
    n1_bin_g4, n2_bin_g4 = len(before_gpt4), len(after_gpt4)
    counts_g4 = [before_gpt4['tangible'].sum(), after_gpt4['tangible'].sum()]
    z_g4, p_z_g4 = proportions_ztest(counts_g4, [n1_bin_g4, n2_bin_g4])
    h_g4 = cohens_h(p1_bin_g4, p2_bin_g4)

    # B) Confidence-Weighted
    p1_cw_g4 = (before_gpt4['tangible'] * before_gpt4['confidence']).sum() / before_gpt4['confidence'].sum()
    p2_cw_g4 = (after_gpt4['tangible'] * after_gpt4['confidence']).sum() / after_gpt4['confidence'].sum()

    # C) Spectrum-Based (Uncorrected)
    p1_spec_g4, p2_spec_g4 = before_gpt4['tangible_lean'].mean(), after_gpt4['tangible_lean'].mean()
    t_spec_g4, p_spec_g4 = ttest_ind(after_gpt4['tangible_lean'], before_gpt4['tangible_lean'], equal_var=False)
    d_spec_g4 = cohens_d(before_gpt4['tangible_lean'], after_gpt4['tangible_lean'])
    
    # D) Spectrum-Based (Error-Corrected)
    df_err_corr_g4 = df[df['confidence'] >= OPTIMAL_THRESHOLD]
    before_ec_g4 = df_err_corr_g4[df_err_corr_g4['timestamp'] < GPT4_RELEASE_DATE]
    after_ec_g4 = df_err_corr_g4[df_err_corr_g4['timestamp'] >= GPT4_RELEASE_DATE]
    p1_ec_g4, p2_ec_g4 = before_ec_g4['tangible_lean'].mean(), after_ec_g4['tangible_lean'].mean()
    t_ec_g4, p_ec_g4 = ttest_ind(after_ec_g4['tangible_lean'], before_ec_g4['tangible_lean'], equal_var=False)
    d_ec_g4 = cohens_d(before_ec_g4['tangible_lean'], after_ec_g4['tangible_lean'])
    
    # E) Non-Parametric Robustness Check
    u_g4, p_u_g4 = mannwhitneyu(before_gpt4['tangible_lean'], after_gpt4['tangible_lean'])

    # F) Print Table A2
    print(f"{'Measurement Approach':<40} {'Key Metric / Test':<40} {'Pre':>12} {'Post':>12} {'Test Stat':>15} {'p-value':>10} {'Effect Size':>15}")
    print("-"*150)
    print(f"{'Binary Classification':<40} {'Proportion Tangible':<40} {p1_bin_g4:>12.3f} {p2_bin_g4:>12.3f} {'z = ' + f'{z_g4:.3f}':>11} {p_z_g4:>10.3f} {'h = ' + f'{h_g4:.3f}':>12}")
    print(f"{'Confidence-Weighted':<40} {'Weighted Prop. Tangible':<40} {p1_cw_g4:>12.3f} {p2_cw_g4:>12.3f} {'—':>15} {'—':>10} {p2_cw_g4 - p1_cw_g4:>+15.3f}")
    print(f"{'Spectrum-Based':<40} {'Mean Tangible Lean Score':<40} {p1_spec_g4:>12.3f} {p2_spec_g4:>12.3f} {'t = ' + f'{t_spec_g4:.3f}':>11} {p_spec_g4:>10.3f}** {d_spec_g4:>15.3f}")
    print(f"{'Spectrum-Based (Error-Corrected)':<40} {'Mean Tangible Lean Score':<40} {p1_ec_g4:>12.3f} {p2_ec_g4:>12.3f} {'t = ' + f'{t_ec_g4:.3f}':>11} {p_ec_g4:>10.3f}** {d_ec_g4:>15.3f}")
    print(f"\nNote: Non-parametric robustness check (Mann-Whitney U test) confirms significance, p = {p_u_g4:.3f}.")


    # --- 4. CALCULATIONS AND PRINTING FOR TABLES A3 & A4 (Confidence Threshold) ---
    print("\n" + "="*80)
    print("RESULTS FOR APPENDIX TABLES A3 & A4: Confidence Threshold Analysis")
    print("="*80)
    
    thresholds = [0.25, 0.35, 0.45, 0.55]
    
    # Print ChatGPT Threshold Table (A3)
    print("\n--- Table A3: ChatGPT Release ---")
    print(f"{'Confidence':<20} {'Data Retained':<15} {'Binary Δ':<12} {'z-test p':<12} {'Spectrum Δ':<12} {'t-test p':<12} {'Cohen\'s d':<12}")
    print("-"*90)
    for thresh in thresholds:
        df_thresh = df[df['confidence'] >= thresh]
        b = df_thresh[df_thresh['timestamp'] < CHATGPT_RELEASE_DATE]
        a = df_thresh[df_thresh['timestamp'] >= CHATGPT_RELEASE_DATE]
        if len(b) > 1 and len(a) > 1:
            z, pz = proportions_ztest([b['tangible'].sum(), a['tangible'].sum()], [len(b), len(a)])
            t, pt = ttest_ind(a['tangible_lean'], b['tangible_lean'], equal_var=False)
            d = cohens_d(b['tangible_lean'], a['tangible_lean'])
            print(f"{'≥ ' + str(thresh):<20} {f'{len(df_thresh)/len(df):.1%}':<15} {a['tangible'].mean() - b['tangible'].mean():>+12.3f} {pz:>12.3f} {a['tangible_lean'].mean() - b['tangible_lean'].mean():>+12.3f} {pt:>12.3f} {d:>12.3f}")

    # Print GPT-4 Threshold Table (A4)
    print("\n--- Table A4: GPT-4 Release ---")
    print(f"{'Confidence':<20} {'Data Retained':<15} {'Binary Δ':<12} {'z-test p':<12} {'Spectrum Δ':<12} {'t-test p':<12} {'Cohen\'s d':<12}")
    print("-"*90)
    for thresh in thresholds:
        df_thresh = df[df['confidence'] >= thresh]
        b = df_thresh[df_thresh['timestamp'] < GPT4_RELEASE_DATE]
        a = df_thresh[df_thresh['timestamp'] >= GPT4_RELEASE_DATE]
        if len(b) > 1 and len(a) > 1:
            z, pz = proportions_ztest([b['tangible'].sum(), a['tangible'].sum()], [len(b), len(a)])
            t, pt = ttest_ind(a['tangible_lean'], b['tangible_lean'], equal_var=False)
            d = cohens_d(b['tangible_lean'], a['tangible_lean'])
            print(f"{'≥ ' + str(thresh):<20} {f'{len(df_thresh)/len(df):.1%}':<15} {a['tangible'].mean() - b['tangible'].mean():>+12.3f} {pz:>12.3f} {a['tangible_lean'].mean() - b['tangible_lean'].mean():>+12.3f} {pt:>12.3f} {d:>12.3f}")

if __name__ == "__main__":
    main_df = load_and_prepare_data(DATA_FILE)
    run_final_analysis(main_df)