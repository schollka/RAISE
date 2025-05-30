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
    
    stableState = [p["aircraftStates"]["stableState"] for p in track]
    prevStableState = [p["aircraftStates"]["prevStableState"] for p in track]
    prevPrevStableState = [p["aircraftStates"]["prevPrevStableState"] for p in track]

    detectedTakeOff = [p["flightEvent"]["detectedTakeOff"] for p in track]
    detectedTouchDown = [p["flightEvent"]["detectedTouchDown"] for p in track]

    flightStateVals, flightStateLabels = mapStates(flightStates)
    stableStateVals, stableStateLabels = mapStates(stableState)
    prevStableStateVals, prevStableStateLabels = mapStates(prevStableState)
    prevPrevStableStateVals, prevPrevStableStateLabels = mapStates(prevPrevStableState)
    detectedTakeOffVals, detectedTakeOffLabels = mapStates(detectedTakeOff)
    detectedTouchDownVals, detectedTouchDownLabels = mapStates(detectedTouchDown)

    fig, axs = plt.subplots(5, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})

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

    # Plot 4: detectedTakeOff and detectedTouchDown
    ax1 = axs[3]
    ax1.step(timestamps, detectedTakeOffVals, where="post", label="detectedTakeOff", color="tab:blue")
    ax1.set_yticks(range(len(detectedTakeOffLabels)))
    ax1.set_yticklabels(detectedTakeOffLabels, color="tab:blue")
    ax1.set_ylabel("detectedTakeOff", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.step(timestamps, detectedTouchDownVals, where="post", label="detectedTouchDown", color="tab:orange")
    ax2.set_yticks(range(len(detectedTouchDownLabels)))
    ax2.set_yticklabels(detectedTouchDownLabels, color="tab:orange")
    ax2.set_ylabel("detectedTouchDown", color="tab:orange")

    # Plot 5: prevStableState and prevPrevStableState
    ax1 = axs[4]
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


def plotScatterPlotAltSpeed(allTracks):
    stateColors = {
        'unknown': 'gray',
        'airborne': 'blue',
        'onGround': 'green',
        'transitionAirGrnd': 'orange',
    }
    from collections import defaultdict
    groupedPoints = defaultdict(lambda: {'alt': [], 'speed': []})

    for track in allTracks:
        for point in track:
            state = point.get('aircraftStates', {}).get('flightState', 'unknown')
            groupedPoints[state]['alt'].append(point.get('altitude', 0))
            groupedPoints[state]['speed'].append(point.get('speed', 0))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for state, data in groupedPoints.items():
        color = stateColors.get(state, 'black')
        plt.scatter(data['alt'], data['speed'], s=10, label=state, color=color, alpha=0.6)

    plt.ylabel("Ground speed [m/s]")
    plt.xlabel("Altitude [m]")
    plt.title("Aircraft States")
    plt.ylim(0, 70)
    plt.xlim(100, 1000)
    plt.legend()
    plt.grid(True)
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