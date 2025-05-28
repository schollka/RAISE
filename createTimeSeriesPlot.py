import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Kategorische Werte in Zahlen umwandeln für die Plots
def mapStates(states):
    unique = {state: i for i, state in enumerate(sorted(set(states)))}
    return [unique[s] for s in states], list(unique.keys())

def plotAltSpeedAndStates(track):
    """
    Erstellt ein kombiniertes Diagramm mit:
    - Geschwindigkeit & Höhe (linke/rechte Achse)
    - Subplots für flightState, flightSubState und receptionState über der Zeit
    """
    # Zeitreihe extrahieren
    timestamps = [p["timestamp"] for p in track]
    speed = [p["groundSpeed"] for p in track]
    altitude = [p["altitude"] for p in track]

    relayed = [p["relayed"] for p in track]
    reducedDataConfidence = [p["reducedDataConfidence"] for p in track]
    

    # FSM-Zustände (als Textlabels)
    aircraftStatesDict = [p.get("aircraftStates", {}) for p in track]
    flightStates = [p.get("flightState", "unknown") for p in aircraftStatesDict]
    
    subStates = [p.get("flightSubState", "none") for p in track]
    stableState = [p["aircraftStates"]["stableState"] for p in track]
    prevStableState = [p["aircraftStates"]["prevStableState"] for p in track]
    prevPrevStableState = [p["aircraftStates"]["prevPrevStableState"] for p in track]

    flightStateVals, flightStateLabels = mapStates(flightStates)

    stableStateVals, stableStateLabels = mapStates(stableState)
    prevStableStateVals, prevStableStateLabels = mapStates(prevStableState)
    prevPrevStableStateVals, prevPrevStableStateLabels = mapStates(prevPrevStableState)

    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    # Plot 1: Geschwindigkeit und Höhe
    ax1 = axs[0]
    ax1.plot(timestamps, speed, label="Speed [m/s]", color="tab:blue")
    ax1.set_ylabel("Speed [m/s]", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(timestamps, altitude, label="Altitude [m]", color="tab:orange")
    ax2.set_ylabel("Altitude [m]", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    ax1.set_title("Time Series Flight Data")

    # Plot 2: flightState
    axs[1].step(timestamps, flightStateVals, where="post", label="flightState", color="tab:green")
    axs[1].set_yticks(range(len(flightStateLabels)))
    axs[1].set_yticklabels(flightStateLabels)
    axs[1].set_ylabel("flightState")

    # Plot 3: stableState
    axs[2].step(timestamps, stableStateVals, where="post", label="stableStates", color="tab:purple")
    axs[2].set_yticks(range(len(stableStateLabels)))
    axs[2].set_yticklabels(stableStateLabels)
    axs[2].set_ylabel("stableState")

    # Plot 4: prevStableState and prevPrevStableState
    ax1 = axs[3]
    ax1.step(timestamps, prevStableStateVals, where="post", label="prevStableState", color="tab:blue")
    ax1.set_yticks(range(len(prevStableStateLabels)))
    ax1.set_yticklabels(prevStableStateLabels, color="tab:blue")
    ax1.set_ylabel("prevStableState", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.step(timestamps, prevPrevStableStateVals, where="post", label="prevPrevStableState", color="tab:orange")
    ax2.set_yticks(range(len(prevPrevStableStateLabels)))
    ax2.set_yticklabels(prevPrevStableStateLabels, color="tab:orange")
    ax2.set_ylabel("prevPrevStableState", color="tab:orange")


    # Achsen formatieren
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axs[1].set_xlabel("Zeit (UTC)")
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()




    '''# Plot 3: distanceToAirport
    axs[2].step(timestamps, distanceToAirport, label="dist", color="tab:red")
    axs[2].set_ylabel("Distance [m]")
    axs[2].set_ylim([0, 1000])'''

    '''
    # Plot 3: flightSubState
    axs[2].step(timestamps, subStateVals, where="post", label="flightSubState", color="tab:red")
    axs[2].set_yticks(range(len(subStateLabels)))
    axs[2].set_yticklabels(subStateLabels)
    axs[2].set_ylabel("subState")
    '''

    '''
    # Plot 4: receptionState
    axs[3].step(timestamps, receptionVals, where="post", label="receptionState", color="tab:purple")
    axs[3].set_yticks(range(len(receptionLabels)))
    axs[3].set_yticklabels(receptionLabels)
    axs[3].set_ylabel("reception")
    '''