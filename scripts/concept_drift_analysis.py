import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    print("Loading data...")

    # Dynamically find the absolute path to ensure execution from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', '03_reviews_final_processed.csv')
    
    # Load data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['cleaned_text', 'date'])

    # Process Dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year

    # Filter years 2019 to 2026 (ignore 2018 as it has only 21 reviews)
    df = df[(df['year'] >= 2019) & (df['year'] <= 2026)]

    # Define keyword groups 
    kw_locker = ['locker', 'lockers', 'θυρίδα', 'θυριδα', 'box', 'skroutz point']
    kw_paper = ['χαρτάκι', 'χαρτακι', 'ειδοποιητήριο', 'ειδοποιητηριο']
    kw_digital = ['sms', 'viber', 'εφαρμογή', 'εφαρμογη', 'app', 'μήνυμα', 'μηνυμα']

    def contains_kw(text, kw_list):
        if not isinstance(text, str): return False
        return any(kw in text for kw in kw_list)

    print("Analyzing keyword frequencies over time...")
    df['has_locker'] = df['cleaned_text'].apply(lambda x: contains_kw(x, kw_locker))
    df['has_paper'] = df['cleaned_text'].apply(lambda x: contains_kw(x, kw_paper))
    df['has_digital'] = df['cleaned_text'].apply(lambda x: contains_kw(x, kw_digital))

    # Calculate percentages per year
    yearly_stats = df.groupby('year').agg(
        total_reviews=('cleaned_text', 'count'),
        locker_count=('has_locker', 'sum'),
        paper_count=('has_paper', 'sum'),
        digital_count=('has_digital', 'sum')
    ).reset_index()

    yearly_stats['Locker/Θυρίδα (%)'] = (yearly_stats['locker_count'] / yearly_stats['total_reviews']) * 100
    yearly_stats['Ειδοποιητήριο/Χαρτάκι (%)'] = (yearly_stats['paper_count'] / yearly_stats['total_reviews']) * 100
    yearly_stats['SMS/Viber/App (%)'] = (yearly_stats['digital_count'] / yearly_stats['total_reviews']) * 100

    print("Generating and saving the line chart...")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style('darkgrid')

    plt.plot(yearly_stats['year'], yearly_stats['Locker/Θυρίδα (%)'], marker='o', linewidth=2.5, color='#e74c3c', label='Locker / Θυρίδα')
    plt.plot(yearly_stats['year'], yearly_stats['Ειδοποιητήριο/Χαρτάκι (%)'], marker='s', linewidth=2.5, color='#95a5a6', label='Χαρτάκι / Ειδοποιητήριο')
    plt.plot(yearly_stats['year'], yearly_stats['SMS/Viber/App (%)'], marker='^', linewidth=2.5, color='#3498db', label='SMS / Viber / App')

    plt.title('Concept Drift: Εξέλιξη Ορολογίας στις Κριτικές (2019-2026)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Έτος', fontsize=12)
    plt.ylabel('Ποσοστό Κριτικών (%)', fontsize=12)
    plt.xticks(yearly_stats['year'])
    plt.legend(title='Θεματολογία', fontsize=11, title_fontsize=12)
    plt.tight_layout()

    # Save image using robust path
    out_path = os.path.join(project_root, 'images', '11_Keyword_Drift.png')
    plt.savefig(out_path, dpi=300)
    print(f'Saved graph to: {out_path}')

if __name__ == '__main__':
    main()
