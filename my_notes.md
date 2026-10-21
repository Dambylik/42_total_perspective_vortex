This project is about teaching a computer to recognize two kinds of movement from recordings of brain activity.

The screen gives an instruction such as:
- “Imagine moving your left hand.”
- “Imagine moving your right hand.”
While the person follows the instruction, the sensors record signals.
Our program receives a short piece of that recording and answers:
“I think this person was imagining their left hand.”
his is a simple brain–computer interface, or BCI: a system that uses brain signals as input to a computer.
64 EEG sensor records a mixture of electrical activity and unwanted signals.
Each sensor produces a sequence of numbers over time. These numbers represent measured electrical potentials. 
A sensor’s recorded signal is called a channel.
What will our program do, step by step?
Recording → cleaning → cutting into windows → extracting clues → making a prediction

Cleaning: make useful patterns easier to see
EEG recordings contain unwanted activity, including effects from blinking, muscles, and electrical equipment.
We use filtering to keep a chosen range of signal frequencies.

Cutting: make manageable examples
A whole recording might last several minutes. We cut it into shorter windows associated with task instructions.
Each window is called an epoch.

Extracting clues: summarize what matters
A recording contains thousands of numbers. We want a smaller collection of useful measurements.
These measurements are called features. For EEG, a useful clue can be how strongly a signal varies during a window.

Making a prediction: choose a class
A classifier is the part that uses those features to choose between the two possible answers.
Features from this window → classifier → “right hand”

A central requirement is to implement a dimensionality-reduction algorithm.
That means turning a large description into a smaller, useful description.
We plan to use Common Spatial Patterns, or CSP.
CSP learns how to combine EEG channels so that differences between the two classes become easier to detect. We then summarize those combined signals into features.

https://archive.physionet.org/physiobank/database/eegmmidb/#:~:text=Experimental%20Protocol

https://mne.tools/stable/generated/mne.channels.make_standard_montage.html



Our goal is to give the computer an EEG window and ask it to choose between two answers. An “experiment” here defines which two answers and which recordings we use.

Run numbers	          T1 means	      T2 means
3, 4, 7, 8, 11, 12	  Left fist	      Right fist
5, 6, 9, 10, 13, 14	  Both fists	    Both feet


Runs	      Condition
3, 7, 11	  Actual movement
4, 8, 12	  Imagined movement

For classification, a common rule is stratification: keep approximately the same class proportions in every fold. If the full set is half left and half right, each fold should also be approximately half left and half right.