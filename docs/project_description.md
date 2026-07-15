# AI in Signal & Audio Processing: Project

This project accounts for 50% of the final grade. This project is in groups of 3 people.
This is the practical synthesis of everything covered in this course: audio processing, feature extraction and  classification pipelines applied to audio. The goal is to use you knowledge on the field and turn it into a practical application, demonstrating a product-oriented mentality. 


## Context

A farmer wants to know when their animals wake up, but is too lazy to wake up early to find out. But the farmer is smart and has a strategy:
place a microphone near the animals and start recording. Once he wakes up, he needs an automatic system to analyze the audio and detect when each animal makes their sound.
The project is about developing such system

## The task
Given a .wav recording of farmyard audio, your system must detect and report sound events belonging to five animal classes: dog, cat, sheep, cow, rooster. A "sound event" is a continuous occurrence of one animal's sound — it has a start time, an end time, and a label. Multiple animals may be vocalizing at the same time, and your system needs to handle that. You also need to consider a sixth class for background noises (everything sound other than the five animals).
Your pipeline takes a full audio file as input and produces two outputs:

1. A structured JSON report, listing every detected event with at minimum its animal label, start time, and end time.
For example:
````
[
    {"event_start": "2.4", "event_end": "3.1", "animal": "sheep"},
    {"event_start": "2.8", "event_end": "5.5", "animal": "dog"} 
]
```` 
2. A visualization of the full audio (waveform or spectrogram) with the detected events marked clearly and legibly, segment by segment, per animal class.

## What you're being given vs. what you need to find

You will receive a seed dataset (a curated subset of [ESC-50](https://github.com/karolpiczak/ESC-50)) covering all five classes. It is intentionally small. Part of your work is deciding whether it's enough, and if not, sourcing, cleaning, and incorporating additional audio — from Freesound, other open datasets, or your own recordings — to build a training set you trust.


## The hard part (and the point of the project)
Classifying a clean 3-second clip of a single animal is not the challenge here. The real challenges are:

- Continuous audio, not pre-cut clips. You decide how to scan a long recording — fixed windows, sliding windows, overlap — and that choice has consequences.
- Overlapping events. More than one animal can be active simultaneously. A model that assumes one label per frame will fail here by design.
- Turning frame-level predictions into events. If your model detects "cow" in seconds 6–6.5 and 7–8, do you treat it as one continuous event with a brief misclassification in the middle, or two distinct events separated by 0.5 seconds? You need to choose a strategy (gap-filling, smoothing, minimum duration thresholds) and you need to be able to explain why you chose it.
- Take into account silence and ambient noise. Most of a real recording is not an animal vocalizing. Your system needs to know when to "say nothing".
- Consider a +/- 500 miliseconds margin for your event detections. Outside those margins we consider that the event detection is not precise.

There is no single correct architecture or post-processing scheme. You are free to use classical ML, neural networks, or a combination — what matters is that your choices are well justified and that fulfill the goal.

## Presentation day
You will receive a previously unseen ~1 minute .wav recording on the day of the presentation. At the beginning of the last session, you will be given the audio and you have to run your pipeline, you will have 15 minutes to make it work and obtain both the resulting JSON output and the visualization. Your solution must be fully ready at the start of the session, those 15 minutes are just so you can run your pipeline. It is meant to show that your system actually works end to end, not just on the data you trained it on.

You will then have 10 minutes to present:

- Your data sourcing and preprocessing decisions
- Your feature representation and model architecture
- Your event-detection and post-processing logic, and why you chose it
- Where your system fails, and why you think it fails there

## What we're grading
This project is 50% of your final grade. We are not grading raw accuracy alone, we are grading the quality of your engineering decisions, your ability to identify and reason about failure modes, and whether your system genuinely works on audio it has never seen.