import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
data = pd.read_pickle("../../data/interim/01_data_processed.pkl" )

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set = data[data["set"] == 1]
set.plot(y=data.columns[:7], figsize=(10,5))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in data["label"].unique():
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    exercise = data[data["label"] == label]
    fig , ax  = plt.subplots()
    plt.plot(exercise[:100]["gyc_y"].reset_index(drop=True), label=label ,color=random.choice(colors)) 
    
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
import random
import matplotlib as mlp
mlp.style.use('seaborn-v0_8')
mlp.rcParams['font.size'] = 12
mlp.rcParams['figure.dpi'] = 150
mlp.rcParams['figure.figsize'] = (10, 5)

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

medium = data[data["category"] == "medium"]
heavy = data[data["category"] == "heavy"]
plt.plot(medium[:100]["gyc_y"].reset_index(drop=True), label="medium")
plt.plot(heavy[:100]["gyc_y"].reset_index(drop=True), label="heavy")

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
data.columns
data.sample(10)

fig, ax = plt.subplots()
# find a data frma whose participant is A and label is squat
data[(data["participant"] == "A") & (data["label"] == "squat")]
category_df = data.query("participant == 'A' & label == 'squat'").reset_index()
category_df.groupby("category")['acc_y'].plot( figsize=(10,5))
plt.legend()
ax.set_xlabel("sample")
ax.set_ylabel("y_acceleration")



# Compare participants
data["category" ].unique()
data["label" ].unique()
comp_df = data.query(" label == 'row'").sort_values('participant').reset_index()
comp_df.groupby("participant")["acc_x"].plot( figsize=(10,5))

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

fig, ax = plt.subplots()
participant = 'A'
label = 'squat'
all_axis_df = data.query(f"participant == '{participant}' & label == '{label}'").reset_index()
all_axis_df.iloc[:,1:4].plot( figsize=(10,5) , linewidth=1)
ax.set_xlabel("sample")
ax.set_ylabel("acceleration and gyroscope")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

for participant in data["participant"].unique():
    for label in data["label"].unique():
        print(participant, label)
        all_axis_df = data.query(f"participant == '{participant}' & label == '{label}'").reset_index()
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df.iloc[:,1:4].plot(ax = ax ,  figsize=(10,5) , linewidth=1)
            ax.set_xlabel("sample")
            ax.set_ylabel("acceleration and gyroscope")
            plt.legend()
            plt.show()



# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
import matplotlib.pyplot as plt

# Define number of participants and labels for subplot grid dimensions
participants = data["participant"].unique()
labels = data["label"].unique()



# Create a grid of subplots, adjust rows and columns as needed


# Loop through participants and labels
for participant in participants:
    for label in labels:
        
        # Query the data for the specific participant and label
        all_axis_df = data.query(f"participant == '{participant}' & label == '{label}'").reset_index()
        
        if len(all_axis_df) > 0:  # Ensure there's data to plot
            fig, axs = plt.subplots(2,sharex=True ,  figsize=(15, 10))
            # Plot data on the corresponding subplot
            all_axis_df.iloc[:, 1:4].plot(ax=axs[0], linewidth=1)
            all_axis_df.iloc[:, 4:7].plot(ax=axs[1], linewidth=1 )

            # Set labels
            axs[0].set_title(f"Participant: {participant}, Label: {label}")
            axs[0].set_xlabel("Sample")
            axs[0].set_ylabel("Acceleration ")
            axs[1].set_ylabel("Gyroscope")
            axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,fancybox=True, shadow=True)	
            axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,fancybox=True, shadow=True)	

            plt.savefig(f"../../reports/figures/{label}_{participant}.png")

# Show the combined figure
plt.show()


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------