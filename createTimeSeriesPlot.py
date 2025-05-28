import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plotFlightProfileFromTrack(track):
    """
    Erstellt ein kombiniertes Diagramm mit:
    - Geschwindigkeit & Höhe (linke/rechte Achse)
    - Subplots für flightState, flightSubState und receptionState über der Zeit
    """
    # Kategorische Werte in Zahlen umwandeln für die Plots
    def mapStates(states):
        unique = {state: i for i, state in enumerate(sorted(set(states)))}
        return [unique[s] for s in states], list(unique.keys())
    
    # Zeitreihe extrahieren
    timestamps = [p["timestamp"] for p in track]
    speed = [p["groundSpeed"] for p in track]
    altitude = [p["altitude"] for p in track]
    climbRate = [p["climbRate"] for p in track]
    distanceToAirport = [p["distanceToAirport"] for p in track]
    relayed = [p["relayed"] for p in track]
    reducedDataConfidence = [p["reducedDataConfidence"] for p in track]
    

    # FSM-Zustände (als Textlabels)
    aircraftStatesDict = [p.get("aircraftStates", {}) for p in track]
    flightStates = [p.get("flightState", "unknown") for p in aircraftStatesDict]
    
    subStates = [p.get("flightSubState", "none") for p in track]
    receptionStates = [p.get("receptionState", "normal") for p in track]
    stableState = [p["aircraftStates"]["stableState"] for p in track]

    flightStateVals, flightStateLabels = mapStates(flightStates)
    realayedVals, relayedStateLabels = mapStates(relayed)
    dataConfVals, dataConfLabels = mapStates(reducedDataConfidence)

    
    subStateVals, subStateLabels = mapStates(subStates)
    receptionVals, receptionLabels = mapStates(receptionStates)
    stableStateVals, stableStateLabels = mapStates(stableState)

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

    ax1.set_title("Flugprofil: Geschwindigkeit & Höhe")

    # Plot 2: flightState
    axs[1].step(timestamps, flightStateVals, where="post", label="flightState", color="tab:green")
    axs[1].set_yticks(range(len(flightStateLabels)))
    axs[1].set_yticklabels(flightStateLabels)
    axs[1].set_ylabel("flightState")

    '''# Plot 3: distanceToAirport
    axs[2].step(timestamps, distanceToAirport, label="dist", color="tab:red")
    axs[2].set_ylabel("Distance [m]")
    axs[2].set_ylim([0, 1000])'''

    # Plot 1: relayed and data confidence
    ax1 = axs[2]
    ax1.step(timestamps, realayedVals, where="post", label="relayedVals", color="tab:blue")
    ax1.set_yticks(range(len(relayedStateLabels)))
    ax1.set_yticklabels(relayedStateLabels, color="tab:blue")
    ax1.set_ylabel("relayed", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.step(timestamps, dataConfVals, where="post", label="dataConfidence", color="tab:orange")
    ax2.set_yticks(range(len(dataConfLabels)))
    ax2.set_yticklabels(dataConfLabels, color="tab:orange")
    ax2.set_ylabel("dataConfidence", color="tab:orange")

    # Plot 3: stableState
    axs[3].step(timestamps, stableStateVals, where="post", label="stableStates", color="tab:purple")
    axs[3].set_yticks(range(len(stableStateLabels)))
    axs[3].set_yticklabels(stableStateLabels)
    axs[3].set_ylabel("stableState")

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
    # Achsen formatieren
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axs[1].set_xlabel("Zeit (UTC)")
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()
