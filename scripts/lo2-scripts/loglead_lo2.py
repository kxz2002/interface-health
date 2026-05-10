import os
import sys
import time
import warnings

import polars as pl
from dotenv import dotenv_values
from loglead import AnomalyDetector
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
from loglead.loaders import LO2Loader

# Suppress ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Adjust full data source
envs = dotenv_values()

# This is the root where the runs are (e.g. "G:/Datasets/lo2-sample/logs")
full_data = envs.get(
    "LOG_DATA_PATH"
)  # If you have defined the log_data_path in loglead, this works
# full_data = "" # if not, add location manually

# List the representations (column names) for anomaly detection. Commented work out too but not used in the analysis so far
items = ["e_words", "e_trigrams", "e_event_drain_id"]

shuffle_train = True
filter_anos = True

# First stage uses the list as is, the second one iterates through it
services = ["client", "code", "key", "refresh-token", "service", "token", "user"]

# ------------------ STAGE 1 ------------------
# Note: This is repeated 10 times in the paper.
# Also, the F1 optimizer isn't enabled. It needs a parameter change in LogLead.

print("---------- LO2: Stage 1 ----------")
frac_data = 1
test_frac = 0.50
print(f"Frac data {frac_data}, test frac {test_frac}")
stime = time.time()

"""
The loader parameters effectively determine the data selection from the total dataset.
In this case we take 100 runs and on each we select a random error 100 times and use all the services.

Params:
n_runs: Number of runs to process.
errors_per_run: Number of errors to include per run.
dup_errors: Whether duplicate errors are allowed across runs.
single_error_type: A specific error type to use exclusively across all runs, or "random" to select one randomly.
single_service: A specific service instead of all to use in the analysis. Options: client, code, key, refresh-token, service, token, user
"""

loader = LO2Loader(filename=full_data, n_runs=100, errors_per_run=1, dup_errors=True)
df = loader.execute()
df = loader.reduce_dataframes(frac=frac_data)
df_seq = loader.df_seq
print("time loaded", time.time() - stime)

df = df.filter(pl.col("m_message").is_not_null())

enhancer = EventLogEnhancer(df)
df = enhancer.length()
stime = time.time()

regexs = [("0", "\d"), ("0", "0+")]
df = enhancer.normalize(regexs, to_lower=True)
print("time normalized", time.time() - stime)
stime = time.time()
df = enhancer.trigrams("e_message_normalized")
print("time trigrams", time.time() - stime)
stime = time.time()
df = enhancer.words("e_message_normalized")
print("time words", time.time() - stime)
stime = time.time()
df = enhancer.parse_drain()
print("time parse", time.time() - stime)
stime = time.time()

seq_enhancer = SequenceEnhancer(df=df, df_seq=df_seq)
# Print for info
print("ano", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"] == False)))
print("normal", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"] == True)))
seq_enhancer.seq_len()

sad = AnomalyDetector(auc_roc=True)
for item in items:
    models_dict = {
        "IsolationForest": {"filter_anos": filter_anos},
        "KMeans": {"filter_anos": filter_anos},
        "RarityModel": {"filter_anos": filter_anos},
        "DT": {},
        "LR": {},
        # "OOVDetector": {"filter_anos":filter_anos}, # Done later with CountVectorizer
    }
    print("-----", item, "-----")
    if "event" in item:
        seq_enhancer.events(item)
    elif item != "m_message":
        seq_enhancer.tokens(item)
    sad.item_list_col = item

    stime = time.time()
    sad.test_train_split(
        seq_enhancer.df_seq,
        test_frac=test_frac,
        shuffle=shuffle_train,
        vectorizer_class=TfidfVectorizer,
    )
    print("time split and prepare:", time.time() - stime)
    sad.evaluate_with_params(models_dict)
    models_dict = {
        "OOVDetector": {"filter_anos": filter_anos},
    }
    stime = time.time()
    sad.test_train_split(
        seq_enhancer.df_seq,
        test_frac=test_frac,
        shuffle=shuffle_train,
        vectorizer_class=CountVectorizer,
    )
    print("time split and prepare:", time.time() - stime)
    sad.evaluate_with_params(models_dict)


# ------------------ STAGE 2 ------------------
error_types = [
    "access_token_authorization_form_401",
    "access_token_auth_header_error_401",
    "access_token_client_id_not_found_404",
    "access_token_client_secret_wrong_401",
    "access_token_form_urlencoded_400",
    "access_token_illegal_grant_type_400",
    "access_token_missing_authorization_header_400",
    "authorization_code_client_id_missing_400",
    "authorization_code_invalid_client_id_404",
    "authorization_code_invalid_password_401",
    "authorization_code_missing_response_type_400",
    "authorization_code_response_not_code_400",
    "code_challenge_invalid_format_pkce_400",
    "code_challenge_too_long_pkce_400",
    "code_challenge_too_short_pkce_400",
    "code_verifier_missing_pkce_400",
    "code_verifier_too_long_pkce_400",
    "code_verifier_too_short_pkce_400",
    "delete_client_404_no_client",
    "delete_service_404_no_service",
    "delete_token_404",
    "delete_user_404_no_user",
    "get_client_404_no_client",
    "get_client_page_400_no_page",
    "get_service_404_no_service",
    "get_service_page_400_no_page",
    "get_token_404",
    "get_token_page_400_no_page",
    "get_user_404_no_user",
    "get_user_page_400_no_page",
    "invalid_code_challenge_method_pkce_400",
    "invalid_code_verifier_format_PKCE_400",
    "register_client_400_clientProfile",
    "register_client_400_clientType",
    "register_client_404_no_user",
    "register_service_400_service_id",
    "register_service_400_service_type",
    "register_service_404_no_user",
    "register_user_400_email_exists",
    "register_user_400_no_password",
    "register_user_400_password_no_match",
    "register_user_400_user_exists",
    "update_client_400_clientProfile",
    "update_client_400_clientType",
    "update_client_404_clientId",
    "update_client_404_ownerId",
    "update_password_400_not_match",
    "update_password_401_wrong_password",
    "update_password_404_user_not_found",
    "update_service_404_service_id",
    "update_service_404_user_id",
    "update_user_404_no_user",
    "verification_failed_pkce_400",
]

print("---------- LO2: Stage 2 ----------")
for et in error_types:
    for service in services:
        print("Test:", et)
        print("Service", service)

        frac_data = 1
        test_frac = 0.50
        print(f"Frac data {frac_data}, test frac {test_frac}")
        stime = time.time()

        # In this case we take 100 runs and on each we select a specific error and service.
        loader = LO2Loader(
            filename=full_data,
            n_runs=100,
            errors_per_run=1,
            dup_errors=True,
            single_error_type=et,
            single_service=service,
        )

        df = loader.execute()
        df = loader.reduce_dataframes(frac=frac_data)
        df_seq = loader.df_seq
        print("time loaded", time.time() - stime)

        df = df.filter(pl.col("m_message").is_not_null())

        enhancer = EventLogEnhancer(df)
        df = enhancer.length()
        stime = time.time()

        regexs = [("0", "\d"), ("0", "0+")]
        df = enhancer.normalize(regexs, to_lower=True)
        print("time normalized", time.time() - stime)
        stime = time.time()

        df = enhancer.words("e_message_normalized")
        print("time words", time.time() - stime)
        stime = time.time()

        seq_enhancer = SequenceEnhancer(df=df, df_seq=df_seq)
        # Print for info
        print(
            "ano",
            len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"] == False)),
        )
        print(
            "normal",
            len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"] == True)),
        )
        seq_enhancer.seq_len()

        sad = AnomalyDetector()
        items = ["e_words"]  # Just use words this time
        for item in items:
            models_dict = {
                "DT": {},  # and just use DT
            }
            print("-----", item, "-----")
            if "event" in item:
                seq_enhancer.events(item)
            elif item != "m_message":
                seq_enhancer.tokens(item)
            sad.item_list_col = item

            stime = time.time()
            sad.test_train_split(
                seq_enhancer.df_seq,
                test_frac=test_frac,
                shuffle=shuffle_train,
                vectorizer_class=TfidfVectorizer,
            )
            print("time split and prepare:", time.time() - stime)
            sad.evaluate_with_params(models_dict)
