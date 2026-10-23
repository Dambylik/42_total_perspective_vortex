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


<Info | 9 non-empty values
 bads: [] - No channels are currently marked as unreliable.
 ch_names: FC5, FC3, FC1, ...- Names identifying the sensors.
 chs: 64 EEG - There are 64 EEG channels.
 custom_ref_applied: False - describe additional EEG processing settings
 dig: 67 items (3 Cardinal, 64 EEG) - Position information: 64 electrodes plus three head reference landmarks.
 highpass: 0.0 Hz - Filter limits recorded in the metadata
 lowpass: 80.0 Hz - Filter limits recorded in the metadata
 meas_date: 2009-08-12 16:15:00 UTC - Recording date and time stored in the file.
 nchan: 64 - There are 64 EEG channels.
 projs: [] - describe additional EEG processing settings
 sfreq: 160.0 Hz - Each channel has 160 measurements per second.
 subject_info: <subject_info | his_id: X, sex: 0, last_name: X> - Participant metadata; the X values are placeholders.
>

Data shape: (channels, samples). Each row is one channel; each column is one measurement time.


Plot 3: frequency content before filtering
raw.compute_psd().plot()
PSD: how power is distributed across frequencies:
PSD means power spectral density. For now, read it as a view of which frequencies contribute strongly to the signal.
- Horizontal axis: frequency in Hz.
- Vertical axis: power per frequency interval, usually displayed on a logarithmic scale.
Imagine a musical chord: the time plot shows the combined sound wave; this plot helps reveal the frequencies making it up.
Here, frequency means how quickly the signal oscillates, not how frequently the computer records samples. Those are different ideas.

What does “10–10 system” mean?
It is a standard way to decide where to place sensors on the head.
People have different head sizes, so researchers use proportions of head measurements, rather than fixed distances in centimetres. “10–10” refers to spacing based on 10% intervals along measured paths over the head.

2. Why keep approximately 7–30 Hz?
| Quantity | Meaning |
|---|---|
| Sampling rate: **160 Hz** | The computer records 160 values each second. |
| Signal frequency: **10 Hz**, for example | A wave in the signal repeats approximately 10 times each second. |

No. sfreq: 160.0 Hz means 160 measurements per second, per channel—not 160 cycles of the EEG wave.
Here, sfreq stands for sampling frequency: how often the equipment records a value.
“Hz” means “times per second.” What happens that many times depends on the context:
Context	What repeats?
Sampling frequency = 160 Hz	Taking a measurement, 160 times per second
Signal frequency = 10 Hz	A wave completing a cycle, 10 times per second


For a simple 10 Hz wave sampled at 160 Hz, we would record 16 measurements during each cycle.
Also, write 160 Hz or 160 measurements per second—“Hz per second” would add “per second” twice.


We need to distinguish two questions:
Question	What we examine
“How large are the recorded voltage fluctuations?”	The signal values, in µV
“How much does each frequency contribute?”	The spectrum: power density, in µV²/Hz

we turn the filtered recording into labeled examples that a model can learn from. This is called epoching. So one epoch is one time window across many channels, not one channel and not one individual measurement.

Unfitered raw : 
 <RawEDF | S001R04.edf, 64 x 60000 (375.0 s), ~29.4 MiB, data loaded>
 the result contains:
- 64 channels
- 60,000 samples per channel
- 375 seconds, because 60000 / 160 = 375
- Approximately 29.4 MiB of loaded data

What does shaep (45, 64, 321) mean?
45 epochs
  × 64 channels per epoch
  × 321 measurements per channel

Each example contains 20,544 measurements (64 × 321). We need to turn those measurements into a few useful features—numbers that help distinguish the classes.

One epoch: 64 channels × 321 samples
                  ↓
                 CSP
                  ↓
Four features describing signal variation
                  ↓
             Classifier
                  ↓
       Predicted label: 1 or 2

Variance measures how spread out a signal’s values are around their average. variance describes one channel’s fluctuations:
To calculate variance:
1. Find the average.
2. Subtract it from each measurement.
3. Square those differences.
4. Take their average.

covariance describes how two channels fluctuate together. CSP uses that relationship to learn useful channel combinations.
To calculate covariance, we:
1. Subtract each channel’s own average.
2. Multiply the two deviations at each matching time.
3. Average those products.
If "-" Their covariance is negative. If "+" Their covariance is positive
If positive and negative products largely cancel, covariance is near zero. That means little linear co-variation, not necessarily no relationship at all.

Each epoch has 64 channels. We calculate covariance for every pair, producing a 64 × 64 covariance matrix:
             Channel 0     Channel 1     ...
Channel 0    variance      covariance
Channel 1    covariance    variance
...
The diagonal contains each channel’s variance—its covariance with itself.


Let’s compare two manually chosen combinations on your real epochs:
C3 + C4 → adds the channels
C3 − C4 → subtracts their shared fluctuations
CSP searches for useful weights across all 64 channels, aiming for different variance between classes, rather than simply large variance overall.

How to express a weighted channel combination using a vector of weights