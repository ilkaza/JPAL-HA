# JPAL-HA
**Justified Human Preferences for Active Learning with Hypothetical Actions (JPAL-HA)** is an human-in-the-loop algorithm for safe agent learning in safety-critical environments. It builds on the **Parenting** algorithm from [(Frye et al., 2019)](https://arxiv.org/pdf/1902.06766.pdf), augmenting it with two novel and generalisable ideas: **Justifications** and **Hypothetical Actions**.

### Files

- _jpal_main.py_: Main module of JPAL-HA including the main logic of the algorithm

- _jpal.py_: Includes
  - all mechanisms of the Agent:
    - ask BTFQs
    - recording
    - ask AFTQs
    - direct policy learning

    with optionally using:
    - Justifications 
    - Hypothetical Actions
    
  - the Model
  
- _human_substitute.py_: A module for substituting the real human with a simulated one by computing the optimal q-values and returning the values and infromation regarding the human

- _client.py_: Includes the Client class for connecting with a server (raspberry pi) which connects with the Thymio robot for the real-word experiment.

- _my_island_navigation.py_: Slightly modified [_island_navigation.py_](https://github.com/deepmind/ai-safety-gridworlds/blob/master/ai_safety_gridworlds/environments/island_navigation.py) from [AI Safety Gridworlds](https://github.com/deepmind/ai-safety-gridworlds) to enable passing a modified configuration of the Island Navigation environment as argument.

- _in_qtable.json_: Stored pretrained q-values which give the optimal policy for the modified Island Navigation environment in the initial modified configuration of the board

### Usage
Coded in Python 3.9 and Pytorch 1.8.1

Dependencies for running the code: **AI Safety Gridworlds**, **Abseil-py**, **Pycolab**

1. Get the [AI Safety Gridworlds](https://github.com/deepmind/ai-safety-gridworlds) and set up the python path to it
2. Install the Abseil package: `pip install absl-py`
3. Download and install [Pycolab](https://github.com/deepmind/pycolab) from source (dependency on AI Safety Gridworld environments): `python setup.py install`
4. For windows: `pip install windows-curses`
5. For running commands from the microphone with speech recognition follow instructions in [Python Voice Assistant Tutorial #1 - Playing Sound with gTTS (Google Text to Speech)](https://www.youtube.com/watch?v=-AzGZ_CHzJk&t=327s):
    - `pip install SpeechRecognition`
    - `pip install gTTS`
    - `pip install playsound`
    - `pip install pyaudio` (if not installed, download appropriate version of [PyAudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and run `pip install <name of .whl file downloaded>`
