<p><a href="https://colab.research.google.com/github/IzaquielCordeiro/NeuralReport/blob/main/NeuralReport.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></p>
<h1 id="neural-report"><strong><code>Neural Report</code></strong></h1>
<p>This is a project from my final paper in Computer Science by <a href="https://www.google.com/maps/place/UFCG+-+Campus+Campina+Grande/@-7.2171368,-35.9097543,15z/data=!4m5!3m4!1s0x0:0xd98e854f0b0d6fe1!8m2!3d-7.2171368!4d-35.9097543">Federal University of Campina Grande</a> A Jupyter Notebook intergrated with Google Drive that seeks and load spike train data acquired from silicon neural probes that measures electrophysiology activity of the brain <em>in-vivo</em>.</p>
<hr>
<p>The current suported data files are:  </p><li><a href="https://klusta.readthedocs.io/en/latest/kwik/">klusta .kwik</a></li>
<li>simple .txt <sub>(containing three rows with space separated values as follow)
</sub></li><ul>
<li>spike time (in seconds)</li>
<li>shank</li>
<li>cluster</li>
</ul>
<p>The current suported visualizations are:</p>
<h3 id="raster-plot">Raster Plot</h3>
<p><img src="https://drive.google.com/uc?export=view&amp;id=1z9pnzYRKJ47jCsBQgPKhaF4yHT2yGgUf" alt="raster plot"></p>
<p>A static Pyplot graph that shows the spike train for <strong>each unique cluster</strong>.<br>
Params:</p>
<ul>
<li>Colorize Shanks: bool (default: True)<br>
<sub>Diferentiate spikes for each shunk.</sub></li>
</ul>
<h3 id="spikes-distribution">Spikes Distribution</h3>
<p><img src="https://drive.google.com/uc?export=view&amp;id=1gNV52gzRw0_L4Ytmleup__-UaC2AZ9lO" alt="raster plot"><br>
A dynamic Plotly graph that shows the distribution of Spikes over time.<br>
Params:</p>
<ul>
<li>Time Frequency Bins: str  (default: “5s”)<br>
<sub>Time window for counting the spikes. e.g.: “5s”, “100ms”, “1min”…</sub></li>
<li>Colorize Shanks: bool (default: True)<br>
<sub>Diferentiate spike counts for each shank</sub></li>
</ul>
<h3 id="firing-rates-trend">Firing Rates Trend</h3>
<p><img src="https://drive.google.com/uc?export=view&amp;id=1qxHSWNuSnQdheIrlXgTx5xqyKDFoMVDK" alt="raster plot"><br>
A dynamic Plotly graph that shows the Trend of Firing Rates over time.<br>
Params:</p>
<ul>
<li>Interval(s): float (default: 1)<br>
<sub>Time interval in Seconds for counting the spikes.</sub></li>
<li>Smoothing Coefficient: int (default: 25)<br>
<sub>Rate to calculate <em>Smoothed Average</em> line using <a href="https://numpy.org/doc/stable/reference/generated/numpy.convolve.html">numpy.convolve</a></sub></li>
</ul>
<h3 id="coefficient-of-variation">Coefficient of Variation</h3>
<p><img src="https://drive.google.com/uc?export=view&amp;id=17YI34K5aSqp4n1jXKNQBeglNBmu3SlIS" alt="raster plot"><br>
A dynamic Plotly graph that shows the Trend of Coefficient of Variation over time.<br>
Params:</p>
<ul>
<li>Interval(s): float (default: 60)<br>
<sub>Time interval in Seconds for counting the spikes.</sub></li>
<li>Fire Rating Interval (s) (default: 10)<br>
<sub>Time interval in Seconds for calculate the Firing Rate</sub></li>
<li>Smoothing Coefficient: int (default: 25)<br>
<sub>Rate to calculate <em>Smoothed Average</em> line using <a href="https://numpy.org/doc/stable/reference/generated/numpy.convolve.html">numpy.convolve</a></sub></li>
</ul>
<h3 id="firing-rates-distribution">Firing Rates Distribution</h3>
<p><img src="https://drive.google.com/uc?export=view&amp;id=1FrfmF4zE_ps7hsM0Z32WMKAfWdjdGk8t" alt="raster plot"><br>
A dynamic Plotly graph that shows the Distribution of Firing Rates by Frequency.<br>
Params:</p>
<ul>
<li>Bins: int (default: 10)<br>
<sub>Number of Bins.&lt;\sub&gt;</li>
<li>Colorize Shanks: bool (default: False)<br>
<sub>Diferentiate firing rates count for each shunk.</sub></li>
</ul>
<hr>
<p>Feel free to contribute :]</p>

