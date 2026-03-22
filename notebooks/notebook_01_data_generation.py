"""
RAW PHARMACY CLAIMS DATA GENERATOR
===================================
Project: Patient Medication Adherence Analysis — Type 2 Diabetes
Author : Dharmanshu Walli

Story baked into data:
- Age 18-35 → lowest PDC (most non-adherent)
- Age 46-60 → highest PDC (most adherent)
- Sitagliptin → worst adherence drug
- Metformin  → best adherence drug
- Rural      → longest refill gaps
- Urban      → best adherence

This script generates RAW CLAIMS — one row per prescription fill.
PDC, MPR, NRx/TRx are NOT pre-calculated. They are derived in notebook 2.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ── CONFIGURATION ──────────────────────────────────────────────
N_PATIENTS        = 500      # number of unique patients
MEASUREMENT_DAYS  = 365      # 1 year observation window
START_DATE        = datetime(2022, 1, 1)
OUTPUT_FILE       = "raw_claims_data.csv"

DRUGS = ["Metformin", "Glipizide", "Sitagliptin"]

# Base fill probability per 30 days (higher = more adherent = more refills)
# Baked-in drug adherence pattern
DRUG_FILL_PROB = {
    "Metformin":   0.82,   # most adherent
    "Glipizide":   0.68,
    "Sitagliptin": 0.58,   # least adherent
}

# Age group fill probability adjustment
AGE_FILL_ADJ = {
    "18-35": -0.22,   # least adherent
    "36-45": -0.10,
    "46-60":  0.08,   # most adherent
    "61-75":  0.04,
    "76-90": -0.03,
}

# Region refill delay adjustment (extra days before refill)
REGION_DELAY = {
    "Urban":      0,    # refills on time
    "Semi-Urban": 6,    # small delay
    "Rural":      18,   # longest delay
}

GENDER_ADJ = {"Male": -0.04, "Female": 0.04}

PRESCRIBERS = [f"DR{str(i).zfill(3)}" for i in range(1, 51)]
PHARMACIES  = [f"PH{str(i).zfill(3)}" for i in range(1, 31)]

# ── HELPER FUNCTIONS ───────────────────────────────────────────

def assign_age_group(age):
    if 18 <= age <= 35: return "18-35"
    elif 36 <= age <= 45: return "36-45"
    elif 46 <= age <= 60: return "46-60"
    elif 61 <= age <= 75: return "61-75"
    else: return "76-90"

def get_fill_probability(drug, age_group, gender, region):
    base = DRUG_FILL_PROB[drug]
    adj  = AGE_FILL_ADJ[age_group] + GENDER_ADJ[gender]
    # Region affects delay not fill probability directly
    prob = base + adj
    return max(0.15, min(0.97, prob))

def generate_days_supply():
    # 65% get 30-day supply, 35% get 90-day supply
    return random.choices([30, 90], weights=[0.65, 0.35])[0]

def generate_patient_claims(patient_id, age, gender, region, drug):
    """Generate all prescription fills for one patient over 1 year"""
    age_group    = assign_age_group(age)
    fill_prob    = get_fill_probability(drug, age_group, gender, region)
    base_delay   = REGION_DELAY[region]
    prescriber   = random.choice(PRESCRIBERS)
    pharmacy     = random.choice(PHARMACIES)

    claims = []
    claim_counter = [0]

    # First fill — index prescription
    first_fill_date = START_DATE + timedelta(days=random.randint(0, 30))
    days_supply     = generate_days_supply()

    claims.append({
        "Patient_ID":       patient_id,
        "Age":              age,
        "Age_Group":        age_group,
        "Gender":           gender,
        "Region":           region,
        "Drug_Name":        drug,
        "Therapy_Class":    "Type 2 Diabetes",
        "Prescriber_ID":    prescriber,
        "Pharmacy_ID":      pharmacy,
        "Fill_Date":        first_fill_date.strftime("%Y-%m-%d"),
        "Days_Supply":      days_supply,
        "Fill_Number":      1,
    })

    # Subsequent fills
    current_date = first_fill_date + timedelta(days=days_supply)
    fill_number  = 2

    while (current_date - START_DATE).days <= MEASUREMENT_DAYS:
        # Does patient refill?
        if random.random() <= fill_prob:
            # On-time or delayed refill based on region + adherence
            if random.random() <= fill_prob:
                # On time or early
                delay = random.randint(-2, 3)
            else:
                # Late refill — gap
                delay = base_delay + random.randint(5, 25)

            fill_date   = current_date + timedelta(days=max(0, delay))
            days_supply = generate_days_supply()

            if (fill_date - START_DATE).days > MEASUREMENT_DAYS:
                break

            claims.append({
                "Patient_ID":       patient_id,
                "Age":              age,
                "Age_Group":        age_group,
                "Gender":           gender,
                "Region":           region,
                "Drug_Name":        drug,
                "Therapy_Class":    "Type 2 Diabetes",
                "Prescriber_ID":    prescriber,
                "Pharmacy_ID":      pharmacy,
                "Fill_Date":        fill_date.strftime("%Y-%m-%d"),
                "Days_Supply":      days_supply,
                "Fill_Number":      fill_number,
            })

            current_date = fill_date + timedelta(days=days_supply)
            fill_number += 1
        else:
            # Patient gaps — skip one refill cycle
            current_date += timedelta(days=days_supply + base_delay + random.randint(10, 40))

    return claims

# ── GENERATE ALL PATIENT CLAIMS ────────────────────────────────

print("Generating raw pharmacy claims data...")
print(f"Patients: {N_PATIENTS} | Measurement Period: {MEASUREMENT_DAYS} days")
print("-" * 50)

all_claims = []
claim_id   = 1

# Patient demographic distribution — weighted toward 18-35 to support story
age_weights = []
for a in range(18, 91):
    if 18 <= a <= 35:   age_weights.append(3.5)
    elif 36 <= a <= 45: age_weights.append(2.0)
    elif 46 <= a <= 60: age_weights.append(2.0)
    elif 61 <= a <= 75: age_weights.append(1.8)
    else:               age_weights.append(1.0)

age_weights = np.array(age_weights) / sum(age_weights)

for i in range(N_PATIENTS):
    patient_id = f"PAT{str(i+1).zfill(4)}"
    age        = int(np.random.choice(range(18, 91), p=age_weights))
    gender     = random.choices(["Male", "Female"], weights=[0.52, 0.48])[0]
    region     = random.choices(["Urban", "Semi-Urban", "Rural"],
                                 weights=[0.45, 0.30, 0.25])[0]
    drug       = random.choices(DRUGS, weights=[0.50, 0.28, 0.22])[0]

    patient_claims = generate_patient_claims(patient_id, age, gender, region, drug)

    for claim in patient_claims:
        claim["Claim_ID"] = f"CLM{str(claim_id).zfill(6)}"
        claim_id += 1
        all_claims.append(claim)

# ── BUILD DATAFRAME ────────────────────────────────────────────

df = pd.DataFrame(all_claims)

# Reorder columns cleanly
col_order = [
    "Claim_ID", "Patient_ID", "Age", "Age_Group", "Gender", "Region",
    "Drug_Name", "Therapy_Class", "Prescriber_ID", "Pharmacy_ID",
    "Fill_Date", "Days_Supply", "Fill_Number"
]
df = df[col_order]
df["Fill_Date"] = pd.to_datetime(df["Fill_Date"])
df = df.sort_values(["Patient_ID", "Fill_Date"]).reset_index(drop=True)

# ── SAVE ───────────────────────────────────────────────────────

df.to_csv(f"/home/claude/{OUTPUT_FILE}", index=False)

# ── VALIDATION SUMMARY ─────────────────────────────────────────

print(f"\n✅ RAW CLAIMS GENERATED SUCCESSFULLY")
print(f"{'='*50}")
print(f"Total Claims (rows)     : {len(df):,}")
print(f"Unique Patients         : {df['Patient_ID'].nunique():,}")
print(f"Avg Claims per Patient  : {len(df)/df['Patient_ID'].nunique():.1f}")
print(f"Date Range              : {df['Fill_Date'].min().date()} → {df['Fill_Date'].max().date()}")
print(f"\nDrug Distribution:")
print(df["Drug_Name"].value_counts().to_string())
print(f"\nRegion Distribution:")
print(df["Region"].value_counts().to_string())
print(f"\nAge Group Distribution:")
print(df["Age_Group"].value_counts().sort_index().to_string())
print(f"\nAvg fills per patient by Age Group (proxy for adherence):")
fills_by_age = df.groupby("Age_Group")["Patient_ID"].count() / \
               df.groupby("Age_Group")["Patient_ID"].nunique()
print(fills_by_age.sort_index().round(1).to_string())
print(f"\n✅ File saved: {OUTPUT_FILE}")
print(f"{'='*50}")
print("\nNext step: Open notebook_02_feature_engineering.ipynb")
print("to calculate PDC, MPR, NRx/TRx, and Refill Gap from these raw claims.")
