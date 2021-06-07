"""
LightGBM
"""
import lightgbm as lgb
import pandas
from utils import DecoratorTimer


class LightGBM_Manager(object):
    model_name = 'LightGBM'

    def __init__(self, cfg):
        self.cfg = cfg
        self.yml_cfg = self.cfg.yml_cfg
        self.model_cfg = self.yml_cfg[self.model_name]
        assert self.cfg.dataset_name == 'wechat1'

    @DecoratorTimer()
    def handle_dataset(self):
        # config
        data_folder_path = self.cfg.data_folder_path
        # columns
        common_columns = ['userid', 'feedid']
        pred_columns = ['read_comment', 'like', 'click_avatar', 'forward']
        action_columns = ['play', 'stay', 'device', 'date_', 'follow', 'favorite', 'comment']
        feed_columns = [
            'authorid', 'videoplayseconds', 'description', 'ocr', 'asr', 'description_char', 'ocr_char',
            'asr_char', 'bgm_song_id', 'bgm_singer_id', 'manual_keyword_list', 'machine_keyword_list',
            'manual_tag_list', 'machine_tag_list', 'feed_embedding'
        ]
        # feat types
        sparse_feat_names = common_columns + \
                            ['follow', 'favorite', 'comment', 'authorid', 'bgm_song_id', 'bgm_singer_id']
        dense_feat_names = ['videoplayseconds', 'play', 'stay']

        # handle
        raw_feed_info = pandas.read_csv(data_folder_path + "/feed_info.csv")
        raw_user_action = pandas.read_csv(data_folder_path + "/user_action.csv")



    def start(self):
        self.handle_dataset()
