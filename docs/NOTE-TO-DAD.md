# Draft note to Dad (edit freely — substance checked, voice yours)

Dad —

Your conundrum was a good one, and Mummy was right: it wanted proper code,
not a chatbot juggling seven converters. So that's what now exists — a small
app called SoloScribe that lives on a Mac. You drop the recording in, tell it
the key and roughly the tempo, paste the chord chart if you have it, and it
gives you back a Guitar Pro file (.gp5) plus the thing you actually asked
ChatGPT for and never got: an audit. It re-plays its own transcription, lines
it up against your recording, and tells you bar by bar how much to trust —
you can listen to the original and its reconstruction side by side. Your
"can't you check it against the original?" question turned out to be the
whole design.

On test phrases where every note is known, it currently gets 90–99% of notes
right, backing band included. A real recording will be somewhat below that —
the report will tell you exactly how far.

There's a page you can look at now — the whole thing explained, with a real
example you can listen to (press play on the report and compare its
reconstruction against the recording):
https://davidgringras.github.io/soloscribe/

And you can try it yourself, today, on any device — upload a clip here:
https://huggingface.co/spaces/DGring/soloscribe

Try it on the clip you mentioned — the 1.5-minute solo — with the key and
rough tempo typed in, and see what comes back. If it embarrasses itself on
real jazz guitar, the report will say precisely how, which is the point.

x