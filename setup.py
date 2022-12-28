#@title Neural Report


import sys
sys.setrecursionlimit(10**6)
import os, fnmatch
from pathlib import Path

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


def drive_search(path=None, format="kwik", follow_links=True, verbose=True, files=False):
    pattern = f"*.{format}"
    total_size = 0
    s = 0
    result = dict()
    try:
        for root, _, f, rootfd in os.fwalk(path, follow_symlinks=follow_links):
            for name in f:
                if fnmatch.fnmatch(name, pattern):
                    if not files: 
                      pdir = os.path.join(Path(root).parent.absolute())
                      if pdir not in result.keys(): result[pdir] = list()
                    else: result[root] = list()
                    s = sum([os.stat(name, dir_fd=rootfd).st_size for name in f if fnmatch.fnmatch(name, pattern)])
                    total_size += s
                    if not files:
                       [result[pdir].append(root.split('/')[-1])]
                    if files: 
                      [result[root].append(name) for name in f if fnmatch.fnmatch(name, pattern)]
                    if verbose and files:
                        print(f"found \033[4m{s}\033[0m consuming ", end="")
                        print(s,  end="")
                        print(f" bytes ({s/1024**2} megabytes) in \033[4m{len(f)}\033[0m non-directory files")
                    break

        if verbose:
          if not files: 
            for dir, files in result.items(): print(f"found \033[4m{len(files)}\033[0m directories containing {format} files in \033[4m{dir}\033[0m")

          print(f"\n================================================================================")
          print(f"total \033[4m{total_size}\033[0m bytes ({s/1024**2} megabytes) founded of {pattern[1:]} files")

    except Exception as e:
        print(e)
    
    finally:
        return result


from klusta.kwik import KwikModel
class DataFile:

  def __init__(self, files, format='kwik', spike_times=[], spike_units=[]):
    if format == 'kwik': self._load_kwik(files)
    elif format == 'txt': self._load_txt(files)
    if spike_times and spike_units: pass


    #TODO
    if len(spike_times) != len(spike_units): raise ValueError("Lists must have equal lenghts")

    # spike_train = pd.DataFrame(columns=["shank_label", "spike_at", "cluster"])
    # self.spike_train = spike_train.append({"shank_label":shank_label, "spike_at":spike_times, "cluster":spike_units}, ignore_index=True)

  
  def _load_kwik(self, logs, group='good'):

    group = 'sua' if group == 'good' else group

    path, logs = logs.popitem()

    shanks = {log.split('.')[0]: KwikModel(os.path.join(path,log)) for log in logs}

    spike_train = pd.DataFrame()
    for i, (shank, kwik_model) in enumerate(shanks.items()):
      groups = kwik_model.cluster_groups
      spikes = pd.DataFrame([(spike_at, cluster) for spike_at, cluster in zip(kwik_model.spike_times, kwik_model.spike_clusters) if groups[cluster] == 'good'], columns=['spike_at','cluster'])

      spike_train = spike_train.append(list(zip(repeat(shank), repeat(i+1), spikes.cluster, spikes.spike_at)))
    spike_train.columns = ["shank_label","shank_id", "cluster", "spike_at"]

    spike_train = spike_train.groupby(['shank_id','cluster']).agg(list)
    spike_train.sort_values('shank_id', inplace=True, ascending=False)
    spike_train['neuron_id'] = list(range(1, 2*len(spike_train.index)+1, 2))
    spike_train = spike_train.explode(['shank_label','spike_at']).reset_index()

    spike_train['datetime'] = spike_train.spike_at.apply(lambda time: datetime.fromtimestamp(time))
    self.spike_train = spike_train.sort_values('spike_at', ascending=True)



  def _load_txt(self, logs):
    path, logs = logs.popitem()
    
    data = {log.split('.')[0]:pd.read_csv(os.path.join(path,log), sep=" ", header=None).T for log in logs}
    
    spike_train = pd.DataFrame()

    for i, (f_name, data) in enumerate(data.items()):
      data.columns = ['spike_at', 'cluster', 'shank_label'] 
      spike_train = spike_train.append(list(zip(data.shank_label.astype(str), data.shank_label.astype(int), data.cluster, data.spike_at)))
    spike_train.columns = ["shank_label","shank_id", "cluster", "spike_at"]

    
    spike_train = spike_train.groupby(['shank_id','cluster']).agg(list)
    spike_train.sort_values('shank_id', inplace=True, ascending=False)
    spike_train['neuron_id'] = list(range(1, 2*len(spike_train.index)+1, 2))
    spike_train = spike_train.explode(['shank_label','spike_at']).reset_index()

    spike_train['datetime'] = spike_train.spike_at.apply(lambda time: datetime.fromtimestamp(time))
    self.spike_train = spike_train.sort_values('spike_at', ascending=True)



from itertools import repeat
import pandas as pd
import numpy as np
import plotly.express as xp
import matplotlib.pyplot as plt

class Animal:

    def __init__(self, datafile):
      spike_train = datafile.spike_train
      self.spike_train = spike_train.sort_values('spike_at', ascending=True)
      self.start, self.end = 0, -1
    
    def get_ifr(self, binsize=10, smoothing=25):      
      ifr_df = pd.DataFrame(columns=['Time'])
      spike_train = self.get_spike_train()
      start, end = spike_train.spike_at.min(), spike_train.spike_at.max()
      shanks = spike_train.shank_label.unique()
      for shank_label in shanks:
        shank_spike_train = spike_train[spike_train.shank_label == shank_label] 
        n_neurons = shank_spike_train.neuron_id.nunique()

        bins = np.arange(start, end, step = binsize)
        count, bins = np.histogram(shank_spike_train.spike_at, bins = bins)
        
        ifr  = pd.Series(count/(binsize*n_neurons), index = bins[:-1])
        ifr_df[shank_label] = ifr.values
        ifr_df['Time'] = ifr.index

      ifr_df["Smoothed Average"] = pd.Series(self._smoothing(ifr_df[shanks].apply('mean', axis=1), smoothing))
      ifr_df['Hours'] = pd.to_datetime(ifr_df.Time, unit='s').dt.strftime('%H:%M:%S:%f')

      return ifr_df.set_index('Time')

    def get_cv(self, ifr_binsize=0.05, step=10, smoothing=25):
      spike_train = self.get_spike_train()

      window = np.arange(spike_train.spike_at.min(), spike_train.spike_at.max(), step = step)
      shanks = spike_train.shank_label.unique()
      cv_df = pd.DataFrame(index = window, columns = shanks)
      ifr = self.get_ifr(binsize=ifr_binsize)

      for t in window:
        for shank in shanks:
          samples = ifr[shank].loc[t:t+step]
          mu = np.mean(samples)
          std = np.std(samples)
          if mu != 0: cv_df[shank].loc[t] = std/mu
          
      cv_df = cv_df.rename_axis('Time').reset_index()

      cv_df["Smoothed Average"] = pd.Series(self._smoothing(cv_df[shanks].apply('mean', axis=1), smoothing))

      cv_df['Hours'] = pd.to_datetime(cv_df.Time, unit='s').dt.strftime('%H:%M:%S:%f')

      return cv_df.drop('Time', axis=1)
    
    def _smoothing(self, y, box_pts):
      box = np.ones(box_pts)/box_pts
      s = np.convolve(y, box, mode='same')
      return s

    def get_spike_train(self):
      return self.spike_train[self.start:self.end]

    def set_time_interval(self, start, end):
      self.start, self.end = start, end
      return self

    def get_fire_rates(self):
      spike_train = self.get_spike_train()
      start, end = spike_train.spike_at.min(), spike_train.spike_at.max()

      duration = end-start

      n_spikes = len(spike_train)

      fire_rates = spike_train.groupby(['shank_label','cluster']).spike_at.count().apply(lambda spike_counts: spike_counts/duration)
      return fire_rates


import ipywidgets as ipw
from ipywidgets import Tab, AppLayout
import logging.config

class Report(Tab):

  animals = {}

  def __init__(self):
    self.dataScreen = _DataScreen()
    self.analysisScreen = _AnalysisScreen()
    super(Report, self).__init__(children=[self.dataScreen,
                                           self.analysisScreen])
    self.set_title(0, "Data")
    self.dataScreen.confirm_btn.on_click(self._confirm_path_data_screen)

    self.set_title(1, 'Analysis')
    
    self._disable_logs()

  def _confirm_path_data_screen(self, btn):
    with self.dataScreen.header_log:
      if not self.dataScreen.animal_multiselect.value or len(self.dataScreen.animal_multiselect.value)==0: print(" No animals selected, please select animals before continue.")
      else: 
        self.dataScreen.selected_animals = self.dataScreen.animal_multiselect.value
        print(f"\033[1;32mAnalysis confirmed on {len(self.dataScreen.selected_animals)} animals.")
        print(f"Fetching data...")
        self.datasets = [drive_search(animal, format=self.dataScreen.file_format, files=True, verbose=False) for animal in self.dataScreen.selected_animals]
        self.animals = {names[-1]:Animal(DataFile(dataset, format=self.dataScreen.file_format)) for dataset in self.datasets for names in [list(dataset.keys())[0].split('/')]}

        print(f'DONE')

        self.analysisScreen.ani_dpdw.options = self.animals
        self.analysisScreen.clear_outputs()


  
  def _disable_logs(self):
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })


class _DataScreen(AppLayout):

  def __init__(self):
    self.btn = ipw.Button(description="btn")

    self.path = '/content/drive/MyDrive/'
    self.selected_path = ''
    self.file_format = 'kwik'
    self.datasets = []
    self.selected_animals = []
    self.valid_formats = [('KWIK', 'kwik'),('TXT', 'txt')]

    self.path_input = ipw.Text(
      value=self.path if self.path else None,
      placeholder='Path to directory',
      description='Path:',
      style={"description_width":"initial"},
      layout = ipw.Layout(width="75%"),
    )

    self.file_format_dpd = ipw.Dropdown(
      options=self.valid_formats,
      value=self.file_format,
      description='File Format:',
      layout=ipw.Layout(width='15%'),
      style={"description":"initial"}
    )


    self.search_btn = ipw.Button(
    description='Search',
    disabled=not self.path,
    tooltip='Run Search',
    layout=ipw.Layout(width="10%")
    ) 

    self.header_log = ipw.Output(layout=ipw.Layout(width="auto",
                            height= "120px",
                            max_height="100px",
                            overflow_y="auto",
                            border="1px solid black"))
    
    self.footer_log = ipw.Output(layout=ipw.Layout(width="75%",
                            height= "auto",
                            max_height="52px",
                            display="flex",
                            flex_flow="row",
                            overflow_y="auto",
                            border="1px solid black"))
    
    self.search_header = ipw.VBox([self.header_log, ipw.HBox([self.path_input, self.file_format_dpd, self.search_btn])])


    self.dataset_select = ipw.Select(options=self.datasets,
              description="Chosse Dataset",
              style={"description_width":"initial"},
              rows=5,
              layout=ipw.Layout(width="auto", height="100px"),
              value=None)
    
    self.animal_multiselect = ipw.SelectMultiple(
                      options=self.dataset_select.value if self.dataset_select.value else [],
                      description='Chosse animals',
                      layout=ipw.Layout(width="100%", height="auto"),
                      style = {"description_width":"initial"})
    
    self.check_all = ipw.widgets.Checkbox(value=False,
                                 description='Select all',
                                 layout=ipw.Layout(width="80%"),
                                 indent=False)

    self.confirm_btn = ipw.Button(description='Confirm',
                         disabled=False,
                         layout=ipw.Layout(height="50px", width="20%", background_color="white"))
    
    self.footer = ipw.HBox([ipw.Label('Selected Animals', style = {'description_width': 'initial'}), self.footer_log, self.confirm_btn], layout=ipw.Layout(height="auto"),style = {'description_width': 'initial'})

    
    super(_DataScreen, self).__init__(header = self.search_header,
                                      left_sidebar = self.dataset_select,
                                      right_sidebar = ipw.HBox([self.animal_multiselect, self.check_all], layout=ipw.Layout(height="100px")),
                                      footer = self.footer,
                                      grid_gap ="5px",
                                      layout=ipw.Layout(height="auto"),
                                      merge=True,
                                      pane_heights=[3,3,2],
                                      juftify_content="center",
                                      align_items = "center")
    
    self.dataset_select.observe(self.on_dataset_select, names="value")
    self.animal_multiselect.observe(self.on_animal_select, names="value")
    self.search_btn.on_click(self.on_search)
    self.check_all.observe(self.select_all, names='value')
    self.path_input.observe(self.set_path, names='value')
    self.file_format_dpd.observe(self.set_file_format, names='value')

  def on_dataset_select(self, change):
    if change['new']:
      logs = list(change['new'].values())[0]
      path = list(change['new'].keys())[0]
      self.selected_path = path
      self.animal_multiselect.options = {log:os.path.join(path, log) for log in logs}

  def set_path(self, change):
    self.path = change['new']
    if self.path: self.search_btn.disabled = False
    else: self.search_btn.disabled = True

  def on_animal_select(self, change): 
    animals = [x for x in change['new']]
    self.footer_log.clear_output()

    with self.footer_log: 
      for animal in animals: print(*animal, sep='', end='\n')
    
  def set_file_format(self, change):
    self.file_format = change['new']


  def on_search(self, btn):
    with self.header_log:
      print('Searching...')
      try:
        self.dataset_select.options = []
        self.dataset_select.options = {key:{key:value} for key, value in drive_search(self.path_input.value, format=self.file_format, files=False).items()}
      except AttributeError:
        self.header_log.clear_output()
        print("\033[91mAn error occurred. Please try again.", end='\n\n')
    
        
  def select_all(self, change):
    if change["type"] == 'change':
      if change["new"]: self.animal_multiselect.value = list(self.animal_multiselect.options.values())
      else: self.animal_multiselect.value = []


from datetime import datetime

class _AnalysisScreen(AppLayout):

  def __init__(self):

    self.n_units_html = ipw.HTML(value= "<h3>Activity Units: 0</h3>")
    self.ani_dpdw = ipw.Dropdown(options=[], value=None, description='Animal:')
    self.time_range_slider = ipw.SelectionRangeSlider(options=[0], index=(0,0), disable=True, description='Time Interval', layout = ipw.Layout(width="500px"))

    self.raster_colorize = ipw.Checkbox(description="Colorize Shanks", value=True, style={"description_width":"initial"})
    self.raster_header = ipw.HBox(children=[ipw.HTML(value="<h2>Raster Plot</h2>"), self.raster_colorize], layout=ipw.Layout(padding="0 0 0 2%", align_items="center", justify_content="space-between"))
    self.raster_PlotBox = _PlotBox(header=self.raster_header)
                                   
    self.spikes_dist_freq = ipw.Combobox(value='5s', placeholder="Ex. 10s", options=['5s', '100ms', '500ms', '15s', '1min'], ensure_option=True, layout=ipw.Layout(width="35%"))
    self.spikes_dist_colorize = ipw.Checkbox(description="Colorize Shanks (may cause unnexpected behavior)", value=False, style={"description_width":"initial"})
    self.spikes_dist_header = ipw.HBox(children=[ipw.HTML(value="<h2>Spikes Distribuition</h2>"), ipw.HBox(children=[ipw.Label(value='Time Frequency Bins'), self.spikes_dist_freq]), self.spikes_dist_colorize], layout=ipw.Layout(padding="0 2% 0 2%", align_items="center", justify_content="space-between"))
    self.spikes_dist_PlotBox = _PlotBox(header=self.spikes_dist_header)

    self.IFR_delta_window = ipw.BoundedFloatText(value=0.5, min=1, max=10000.0, step=0.1, description='', layout=ipw.Layout(width="30%"))
    self.IFR_smoothing_coef = ipw.BoundedIntText(value=25, min=1, max=100, step=1, description='', layout=ipw.Layout(width="30%"))
    self.IFR_header = ipw.HBox(children=[ipw.HTML(value="<h2>Instantaneous Fire Rates Trend</h2>"), ipw.HBox(children=[ipw.Label(value='Interval (s)'), self.IFR_delta_window]), ipw.HBox(children=[ipw.Label(value='Smoothing Coeficient'), self.IFR_smoothing_coef])], layout=ipw.Layout(padding="0 2% 0 2%", align_items="center", justify_content="space-between"))
    self.IFR_PlotBox =  _PlotBox(header=self.IFR_header)
   
    self.CV_delta_window = ipw.BoundedFloatText(value=60, min=0.5, max=10000.0, step=0.5, description='', layout=ipw.Layout(width="30%"))
    self.CV_ifr_delta_window = ipw.BoundedFloatText(value=10, min=.05, max=100.0, step=.05, description='', layout=ipw.Layout(width="30%"))
    self.CV_smoothing_coef = ipw.BoundedIntText(value=25, min=1, max=100, step=1, description='', layout=ipw.Layout(width="30%"))
    self.CV_header = ipw.HBox(children=[ipw.HTML(value="<h2>Coefficient of Variation Trend</h2>"), ipw.HBox(children=[ipw.Label(value='Interval (s)'), self.CV_delta_window]), ipw.HBox(children=[ipw.Label(value='IFR Interval (s)'), self.CV_ifr_delta_window]), ipw.HBox(children=[ipw.Label(value='Smoothing Coeficient'), self.CV_smoothing_coef])], layout=ipw.Layout(padding="0 2% 0 2%", align_items="center", justify_content="space-between"))
    self.CV_PlotBox = _PlotBox(header=self.CV_header)

    self.fire_rates_histplot_nbins = ipw.BoundedIntText(value=10, min=0, max=500, step=1, description='', layout=ipw.Layout(width="45%"))
    self.fire_rates_colorize = ipw.Checkbox(description="Colorize Shanks", value=False, style={"description_width":"initial"})
    self.fire_rates_histplot_header = ipw.HBox(children=[ipw.HTML(value="<h2>Fire Rates Histogram</h2>"), ipw.HBox(children=[ipw.Label(value='Bins'), self.fire_rates_histplot_nbins]), self.fire_rates_colorize], layout=ipw.Layout(padding="0 2% 0 2%", align_items="center", justify_content="space-between"))
    self.fire_rates_PlotBox = _PlotBox(header=self.fire_rates_histplot_header)  
    
    self.graphs = ipw.VBox(children=[self.raster_PlotBox, self.spikes_dist_PlotBox, self.IFR_PlotBox, self.CV_PlotBox, self.fire_rates_PlotBox], layout=ipw.Layout(width="100%", border="1px solid gray"))

    self.confirm_btn = ipw.Button(description='Confirm', width='100%')
    self.confirm_btn.on_click(lambda click: self._refresh_views(
        animal=self.ani_dpdw.value.set_time_interval(self.time_range_slider.index[0], self.time_range_slider.index[1]),
        ifr_delta=self.IFR_delta_window.value,
        ifr_smooth_coef=self.IFR_smoothing_coef.value,
        cv_delta=self.CV_delta_window.value,
        cv_ifr_binsize=self.CV_ifr_delta_window.value,
        cv_smooth_coef=self.IFR_smoothing_coef.value,
        spikes_dist_freq =  self.spikes_dist_freq.value,
        spikes_dist_color = self.spikes_dist_colorize.value,
        raster_color = self.raster_colorize.value,
        fire_rates_histplot_nbins = self.fire_rates_histplot_nbins.value,
        fire_rates_histplot_color = self.fire_rates_colorize.value,
        ))

                          
    self.menu = ipw.HBox(children=[self.ani_dpdw, self.time_range_slider, self.n_units_html, self.confirm_btn], layout=ipw.Layout(align_items="center", justify_content="space-between"))


    super(_AnalysisScreen, self).__init__(center=ipw.VBox(children=[self.menu, self.graphs]),
                       justify_content='center', align_items='center')
    
    
    self.ani_dpdw.observe(lambda change: self.update_header(change['new']), names='value')

  def update_header(self, animal):
    spike_train = animal.get_spike_train()
    self.n_units_html.value = f"<h3>Activity Units: {spike_train.neuron_id.nunique()}</h3>"
    self.time_range_slider.options = spike_train.datetime.dt.strftime("%H:%M:%S")
    self.time_range_slider.index = (0, len(spike_train)-1)
    self.time_range_slider.disabled = False

  def _fig_graph(self, df, title='Title', ylabel=None):
    columns = sorted([col for col in df.columns if col != 'Smoothed Average'])+['Smoothed Average']
    fig = xp.line(df[columns])

    fig.update_layout(title=title,
                    xaxis_title='Time',
                    yaxis_title=ylabel,
                    legend_traceorder="normal",
                    hovermode="x unified")
    fig.update_traces(mode="lines", hovertemplate=None)
    return fig

  def _get_spikes_dist_fig(self, spike_train, freq, color=False):
    
    temp = spike_train.sort_values('datetime')
    temp = temp.groupby([pd.Grouper(key='datetime', freq=freq), 'shank_id', 'shank_label','neuron_id']).count().reset_index()
    temp['time'] = pd.to_datetime(temp['datetime'], unit='f').dt.strftime('%H:%M:%S:%f')

    fig =  xp.histogram(x=temp.time, y=temp.spike_at, color=temp.shank_label if color else None, histfunc='sum').update_layout(
        title_text='Spikes Dist Count by Fequency', 
        xaxis_title_text='Time', 
        yaxis_title_text='Count',
        hovermode="x unified"
    )
    return fig.update_traces(hovertemplate='Spikes: %{y}')

  def _get_raster_fig(self, spike_train, color):
    activities = spike_train.sort_values('shank_id')
    height = activities.neuron_id.nunique()//10
    fig, ax = plt.subplots(figsize=(42, height if height > 10 else 10))
    color = 'shank_id' if color else None
    
    raster_fig = ax.scatter(data=activities, x='spike_at', y='neuron_id', c=color, cmap='Dark2', marker='|')
    ax.set_xticklabels(pd.to_datetime(pd.Series(ax.get_xticks()), unit='s').dt.strftime('%H:%M:%S').values)
    ax.tick_params(axis='x', which='major', labelsize=20)
    try:
      plt.legend(handles=raster_fig.legend_elements()[0], labels=activities.shank_label.unique().tolist(), fontsize='x-large', markerscale=2, bbox_to_anchor=(1, 1), fancybox=True, shadow=True, prop={'size': 15})
    except ValueError:  
      pass
    return fig

  def _get_fire_rates_histplot(self, fire_rates, nbins=10, color=None):
      color = fire_rates.shank_label if color else None
      fig = xp.histogram(x=fire_rates.spike_at,
                      color=color,
                      nbins=nbins)
      
      fig.update_layout(
        title_text='FR dist', 
        xaxis_title_text='Frequency', 
        yaxis_title_text='#$Spikes', 
        bargap=0.1,
        hovermode="x unified"
      )

      return fig.update_traces(hovertemplate='Fire Rates: %{y}')


  def _refresh_views(self, **args):
      if not args.get("animal"): return
      animal = args.get("animal")
      spike_train = animal.get_spike_train()
      self.clear_outputs()
      
      with self.raster_PlotBox.get_output():
        print()
        fig = self._get_raster_fig(spike_train, args.get('raster_color'))
        display(fig)
        plt.close(fig)
      with self.spikes_dist_PlotBox.get_output():
        print()
        color = 'shank_label' if args.get('spikes_dist_color') else None
        spikes_dist_fig = self._get_spikes_dist_fig(spike_train, freq=args.get('spikes_dist_freq'), color=color)
        display(spikes_dist_fig)
      with self.IFR_PlotBox.get_output():
        print()
        ifr_df = animal.get_ifr(binsize=args.get("ifr_delta"), smoothing=args.get("ifr_smooth_coef")).set_index("Hours")
        ifr_fig = self._fig_graph(ifr_df.fillna(0), title=f'Instantaneous Fire Rate', ylabel='Count')
        display(ifr_fig)
      with self.CV_PlotBox.get_output():
        print()
        cv_df = animal.get_cv(step = args.get("cv_delta"), ifr_binsize=args.get("cv_ifr_binsize"), smoothing=args.get("cv_smooth_coef")).set_index("Hours")
        cv_fig = self._fig_graph(cv_df.fillna(0), title=f'Coefficient of Variation', ylabel='CV')
        display(cv_fig)
      with self.fire_rates_PlotBox.get_output():
        print()
        fig = self._get_fire_rates_histplot(animal.get_fire_rates().reset_index(), nbins=args.get('fire_rates_histplot_nbins'), color=args.get('fire_rates_histplot_color'))
        display(fig)

  def clear_outputs(self):
    for box in self.graphs.children: box.clear_output()


from ipywidgets import VBox
class _PlotBox(VBox):

  def __init__(self, header):
    self.header = header
    self.output = ipw.Output()
    super(_PlotBox, self).__init__(children=[self.header, self.output], layout=ipw.Layout(border="1px solid gray"))
  
  def get_output(self):
    return self.output
  
  def clear_output(self):
    self.output.clear_output()

#/content/drive/MyDrive/UFCG