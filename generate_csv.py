import csv
import random

# List of random names to choose from
names = ['Alice', 'Bob', 'Charlie', 'David', 'Emily', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack', 'Katie', 'Liam', 'Mia', 'Noah', 'Olivia', 'Peter', 'Quinn', 'Riley', 'Sofia', 'Tom', 'Violet', 'William', 'Xander', 'Yara', 'Zachary']

# List of infectious disease modelling related words
words = ['Agent-based modeling', 'Antibody', 'Asymptomatic transmission', 'Attack rate', 'Basic reproduction number', 'Bayesian inference', 'Biosecurity', 'Case fatality rate', 'Cluster', 'Compartmental models', 'Contact network', 'Contact rate', 'Contact structure', 'Contact tracing', 'Contagion', 'Control measures', 'Data assimilation', 'Data-driven modeling', 'Diagnostic testing', 'Disease progression', 'Dynamic transmission model', 'Effective reproduction number', 'Epidemic', 'Epidemiology', 'Exponential growth', 'Genetic diversity', 'Global health', 'Herd immunity', 'Immune system', 'Immunity', 'Immunity passport', 'Infectious disease surveillance', 'Infectiousness', 'Infection', 'Infection control', 'Infectivity', 'Incidence', 'Incubation period', 'Influenza', 'Isolation', 'Mathematical modeling', 'Model calibration', 'Mortality', 'Mutation', 'Non-pharmaceutical interventions', 'Outbreak', 'Pandemic', 'Pandemic preparedness', 'Parameter estimation', 'Pathogen', 'Population dynamics', 'Probability', 'Public health', 'Public health intervention', 'Quarantine', 'Reinfection', 'Reproduction number', 'Risk', 'Seasonal influenza', 'Seasonality', 'Severity', 'SIR model', 'Simulation model', 'Spillover event', 'Stochastic simulation', 'Strain', 'Super-spreader', 'Testing', 'Thresholds', 'Transmission', 'Treatment', 'Vector', 'Viral load', 'Virus']

# Open the CSV file for writing
with open('infectious_disease_modelling.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Name', 'Words related to infectious disease modelling'])

    # Write 1000 rows of random names and words
    for i in range(1000):
        name = random.choice(names)
        words_selected = random.sample(words, 50)
        row = [name] + words_selected
        writer.writerow(row)

print('CSV file generated successfully!')
