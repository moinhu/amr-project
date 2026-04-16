import numpy as np
import pandas as pd

np.random.seed(42)

def generate_resistant_sample():
    return {
        'blaTEM': np.random.binomial(1, 0.65),
        'blaKPC': np.random.binomial(1, 0.50),
        'mecA': np.random.binomial(1, 0.45),
        'vanA': np.random.binomial(1, 0.35),
        'tetM': np.random.binomial(1, 0.55),
        'sul1': np.random.binomial(1, 0.60),
        'aac6': np.random.binomial(1, 0.50),
        'int1_integron': np.random.binomial(1, 0.60),
        'plasmid_count': np.random.poisson(2) + np.random.randint(0, 2),
        'transposon_present': np.random.binomial(1, 0.55),
        'virulence_score': np.random.normal(6.5, 2.0),
        'gc_content': np.random.normal(51, 5),
        'genome_size_mbp': np.random.normal(5.0, 1.0),
        'mutation_gyrA': np.random.binomial(1, 0.70),
        'mutation_parC': np.random.binomial(1, 0.55),
        'efflux_pump_score': np.random.normal(6.0, 1.8),
        'outer_membrane_loss': np.random.binomial(1, 0.40),
        'biofilm_formation': np.random.binomial(1, 0.55),
        'species_code': np.random.choice([1,2,3,4]),
        'snp_count': int(np.random.normal(70, 30)),
        'noise_feature_1': np.random.normal(0, 1),
        'noise_feature_2': np.random.uniform(0, 1),
        'label': 1
    }

def generate_susceptible_sample():
    return {
        'blaTEM': np.random.binomial(1, 0.30),
        'blaKPC': np.random.binomial(1, 0.15),
        'mecA': np.random.binomial(1, 0.20),
        'vanA': np.random.binomial(1, 0.15),
        'tetM': np.random.binomial(1, 0.30),
        'sul1': np.random.binomial(1, 0.35),
        'aac6': np.random.binomial(1, 0.25),
        'int1_integron': np.random.binomial(1, 0.35),
        'plasmid_count': np.random.poisson(1.2),
        'transposon_present': np.random.binomial(1, 0.25),
        'virulence_score': np.random.normal(4.5, 2.0),
        'gc_content': np.random.normal(50, 5),
        'genome_size_mbp': np.random.normal(4.8, 1.0),
        'mutation_gyrA': np.random.binomial(1, 0.30),
        'mutation_parC': np.random.binomial(1, 0.25),
        'efflux_pump_score': np.random.normal(4.5, 1.8),
        'outer_membrane_loss': np.random.binomial(1, 0.20),
        'biofilm_formation': np.random.binomial(1, 0.35),
        'species_code': np.random.choice([1,2,3,4]),
        'snp_count': int(np.random.normal(45, 25)),
        'noise_feature_1': np.random.normal(0, 1),
        'noise_feature_2': np.random.uniform(0, 1),
        'label': 0
    }

def introduce_label_noise(df, noise_rate=0.05):
    n_flip = int(len(df) * noise_rate)
    flip_indices = np.random.choice(df.index, n_flip, replace=False)
    df.loc[flip_indices, 'label'] = 1 - df.loc[flip_indices, 'label']
    return df

def create_dataset():
    n_resistant, n_susceptible = 700, 500

    data = [generate_resistant_sample() for _ in range(n_resistant)] + \
           [generate_susceptible_sample() for _ in range(n_susceptible)]

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add label noise
    df = introduce_label_noise(df, noise_rate=0.05)

    df.to_csv("data/amr_dataset.csv", index=False)
    return df