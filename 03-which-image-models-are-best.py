import pandas as pd
import plotly.express as px

subdir_path = 'pytorch-image-models/results'
df_results = pd.read_csv(f'{subdir_path}/results-imagenet.csv')

def get_data(part, col):
    df = pd.read_csv(f'{subdir_path}/benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').merge(df_results, on='model')
    df['secs'] = 1. / df[col]
    df['family'] = df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')
    df = df[~df.model.str.endswith('gn')]
    df.loc[df.model.str.contains('in22'),'family'] = df.loc[df.model.str.contains('in22'),'family'] + '_in22'
    df.loc[df.model.str.contains('resnet.*d'),'family'] = df.loc[df.model.str.contains('resnet.*d'),'family'] + 'd'
    return df[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg')]

df = get_data('infer', 'infer_samples_per_sec')

w,h = 1000,800

def show_all(df, title, size):
    return px.scatter(df, width=w, height=h, size=df[size]**2, title=title,
        x='secs',  y='top1', log_x=True, color='family', hover_name='model', hover_data=[size])


show_all(df, 'Inference', 'infer_img_size').show()

subs = 'levit|resnetd?|regnetx|vgg|convnext.*|efficientnetv2|beit'

def show_subs(df, title, size):
    df_subs = df[df.family.str.fullmatch(subs)]
    return px.scatter(df_subs, width=w, height=h, size=df_subs[size]**2, title=title,
        trendline="ols", trendline_options={'log_x':True},
        x='secs',  y='top1', log_x=True, color='family', hover_name='model', hover_data=[size])

show_subs(df, 'Inference', 'infer_img_size').show()

px.scatter(df, width=w, height=h,
    x='param_count_x',  y='secs', log_x=True, log_y=True, color='infer_img_size',
    hover_name='model', hover_data=['infer_samples_per_sec', 'family']
).show()

tdf = get_data('train', 'train_samples_per_sec')

show_all(tdf, 'Training', 'train_img_size').show()

show_subs(tdf, 'Training', 'train_img_size').show()
