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

One honest correction first: something commercial does now exist. Klangio's
"Guitar2Tabs" (klang.io) takes audio and spits out a GP5 directly — about
$35 for the first year. There's also a free two-step route: Spotify's free
tool at basicpitch.spotify.com turns audio into MIDI in the browser, and
TuxGuitar (free) opens MIDI and saves as .gp5. Worth trying on your clip for
comparison. The catch, and the reason I still built ours: the one independent
review of these tools found the output rougher than transcribing by hand, and
none of them will tell you *where* they went wrong — you get a chart that's
20% nonsense with no map of which 20%. Ours measures itself, handles swing
the way real charts do (straight eighths marked "Swing", not triplet soup),
and knows the difference between a swung line and a genuine triplet.

On test phrases where every note is known, it currently gets 90–99% of notes
right, backing band included. A real recording will be somewhat below that —
the report will tell you exactly how far.

Send me the clip (the 1.5-minute solo you mentioned), the key, the tempo if
you know it, and a photo of the chart, and I'll run it and send you back the
.gp5, a PDF of the notation, and the report. If it embarrasses itself on
real jazz guitar, we'll know precisely how, which is the point.

x
