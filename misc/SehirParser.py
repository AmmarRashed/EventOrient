import psycopg2

import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import warnings

import unicodedata

from joblib import Parallel, delayed

SEHIR_USER_COLUMNS = ['First Name', 'Last Name', 'Primary Email']
TWITTER_COLS = {'id': 'GUID', 'name': 'twitter_name','screen_name': 'twitter_screen_name'}
DEFAULT_COLS = {"id":"id", "name":"twitter_name", "screen_name":"twitter_screen_name"}
class SehirParser:
    def __init__(self, sehir_contacts_path, db_path=None, cols=DEFAULT_COLS,
                 sehir_usecols=SEHIR_USER_COLUMNS,
                 encoding="ISO-8859-1",
                 ):
        self.cols = cols
        self.db_path = db_path
        self.sehir_directory = pd.read_csv(sehir_contacts_path,
                                      encoding=encoding,
                                      usecols=sehir_usecols).dropna()
        self.fullnames = list()
        self.twitter_users_count = 0
        self.counter = 0
        self.sql_dbname, self.sqlhost, self.sqluser, self.sqlpass = None, None, None, None
        self.twitter_users = None
        self.user_connections = None
        self.twitter_user_by_screen_name = None
        self.update_fullnames(self.sehir_directory)
        self.connect_db(db_path)

    def update_fullnames(self, contacts=None):
        if contacts is None:
            contacts = self.sehir_directory
        self.fullnames = [' '.join(first_last_name).lower()
                        for first_last_name in contacts[['First Name', 'Last Name']].values]
        return self.fullnames

    @staticmethod
    def get_matches_edit_distance(item, choices, limit, scorer=fuzz.token_sort_ratio):
        return process.extract(item, choices, limit=limit, scorer=scorer)


    def matching(self, twitter_screen_name, limit):
        try:
            twitter_name = self.twitter_user_by_screen_name.loc[twitter_screen_name]['cleaned_twitter_name']
        except KeyError:
            warnings.warn("NaN name")
            return []
        if type(twitter_name) != str and len(twitter_name) > 1:  ## There are accounts with exactly the same name
            twitter_name = list(twitter_name)[0]
        sehir_matches = self.get_matches_edit_distance(twitter_name, self.fullnames, limit)
        self.counter += 1
        if self.counter % 100 == 0:
            print(self.counter, "out of ", self.twitter_users_count)
        return sehir_matches

    def filter_matches_by_threshold(self, matches_dict, threshold=85):
        filtered_dict = dict()
        for twitter_screen_name, matches in matches_dict.items():
            filtered = [(match, score) for match, score in matches if score > threshold]

            if filtered:
                filtered_dict[twitter_screen_name] = filtered

        return filtered_dict

    def get_matches_dataframe(self, threshold, limit):
        self.twitter_user_by_screen_name = self.twitter_users.set_index(self.cols['screen_name'])
        twitter_names = self.twitter_users[self.cols['screen_name']]
        sehir_matches = Parallel(n_jobs=-1)(delayed(self.matching)(twitter_screen_name, limit) for twitter_screen_name in
                            twitter_names)
        matches = dict()
        for i in range(len(twitter_names)):
            if len(sehir_matches[i]) > 0:
                matches[twitter_names[i]] = sehir_matches[i]

        filtered_matches = self.filter_matches_by_threshold(matches, threshold=threshold)
        screen_names = filtered_matches.keys()
        return pd.DataFrame({self.cols['screen_name']: list(screen_names),
                             'sehir_matches': [filtered_matches[screen_name] for screen_name in screen_names]})

    @staticmethod
    def clean(name, min_len=5, junk_replacement=''):
        try:
            cleaned = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').lower().decode("ascii")
        except TypeError:
            return junk_replacement
        if len(cleaned) < min_len:
            return junk_replacement
        return cleaned

    def connect_db(self, sql_dbname=None, sqlhost=None, sqluser=None, sqlpass=None, cols=TWITTER_COLS):
        if self.db_path is None:

            if self.sql_dbname is None:
                self.sql_dbname = sql_dbname

            if self.sqlhost is None:
                self.sqlhost = sqlhost

            if self.sqluser is None:
                self.sqluser = sqluser

            if self.sqlpass is None:
                self.sqlpass = sqlpass

            if None in [self.sql_dbname, self.sqlhost, self.sqluser, self.sqlpass]:
                warnings.warn("Missing sql credentials. Call connect_db with the right credentials")
                return
            connection = psycopg2.connect('dbname=%s host=%s user=%s password=%s'%(self.sql_dbname, self.sqlhost, self.sqluser, self.sqlpass))

            self.twitter_users = pd.read_sql("SELECT * FROM twitter_user", connection)\
                .rename(columns={'id': cols["id"],
                                 'name': cols["name"],
                                 'description': 'profile_description',
                                 'screen_name': cols['screen_name']})
            self.cols = cols
            self.user_connections = pd.read_sql("SELECT * FROM twitter_connection", connection).drop('id', axis=1)
        else:
            try:
                self.twitter_users = pd.read_csv(self.db_path).rename(columns=cols)
            except:
                df = pd.read_csv(open("datasets/tw_users.csv", 'rU'), encoding='utf-8', engine='c')
                df["GUID"] = df["GUID.1"]
                self.twitter_users = df.drop("GUID.1", axis=1)
                self.twitter_users.rename(columns=cols, inplace=True)

            self.clean_twitter_users()

        self.twitter_users_count = len(self.twitter_users)

    def clean_twitter_users(self, min_len=5, junk_replacement=np.NaN):
        self.twitter_users["cleaned_twitter_name"] = self.twitter_users[self.cols["name"]].apply(
            lambda x: self.clean(x, min_len, junk_replacement))
        if self.db_path is None:
            self.twitter_users = self.twitter_users[self.twitter_users.full_name != np.NaN]
        else:
            self.twitter_users = self.twitter_users[self.twitter_users.cleaned_twitter_name != np.NaN]

    def get_sehir_matches_df(self, threshold=85, limit=1):
        self.counter = 0
        if self.db_path is None and None in [self.sql_dbname, self.sqlhost, self.sqluser, self.sqlpass]:
            warnings.warn("Missing sql credentials. Call connect_db with the right credentials")
            pass
        sehir_matches_df = self.get_matches_dataframe(threshold=threshold, limit=limit)
        merged = sehir_matches_df.merge(self.twitter_users,
                                        left_on=self.cols["screen_name"], right_on=self.cols["screen_name"])

        return merged, sehir_matches_df

# cols = {"id":"id", "name":"full_name", "screen_name":"full_name"}
# s = SehirParser('datasets/contacts.csv', "datasets/fb_users_toy.csv", cols)
# merged, sehir_matches_df = s.get_sehir_matches_df()


# cols = {"id":"GUID", "name":"cleaned_twitter_name", "screen_name":"twitter_screen_name"}
# fb_sp = SehirParser('datasets/contacts.csv', "datasets/tw_users.csv", cols)