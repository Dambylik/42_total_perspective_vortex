from preprocess import load_raw

raw = load_raw(subject=1, runs=[3])

print(raw)
print("Sampling rate:", raw.info["sfreq"])
print("Channel names:", raw.ch_names)

data = raw.get_data(picks=[0, 1], start=0, stop=10)

print("Shape:", data.shape)
print(data * 1_000_000)


for annotation in raw.annotations[:10]:
    print(
        "Start:", annotation["onset"],
        "Duration:", annotation["duration"],
        "Code:", annotation["description"],
    )