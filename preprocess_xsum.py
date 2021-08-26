"""
Preprocess the XSUM dataset

    There are several noisy training instances which do not contain any words in pre-defined vocabulary of NTM.
    We remove these instances.

    Here are the details about these removed instance:
    - instance #37993:
    input: Here are our favourites:
    target: On Monday, we asked for you to send us your favourite shop pun names.
    - instance #47104:
    input: Here are some of the Ethiopian runner's greatest feats.
    target: Haile Gebrselassie has announced his retirement from competitive running, bringing to an end a 25-year career in which he claimed two Olympic gold medals, eight World Championship victories and set 27 world records.
    - instance #71767:
    input: JANUARYFEBRUARYMARCHAPRILMAYJUNE
    target: As 2015 draws to an end, we take a look back at some of the major stories of the year, along with others that proved popular with readers.
    - instance #94109:
    input: Donegal 1-14 1-12 MayoDown 0-06 0-22 KerryDerry 2-12 1-18 GalwayLaois 0-14 1-14 TyroneMeath 1-13 1-20 CavanAntrim 2-14 0-09 Leitrim
    target: FOOTBALL LEAGUE RESULTS
    - instance #95592:
    input: KERRY 1-13 1-8 DONEGALMONAGHAN 1-12 2-11 MAYOROSCOMMON 1-12 0-6 DOWNFERMANAGH 1-17 0-10 LAOISLONDON 0-11 1-11 ANTRIMAllianz Hurling LeagueWESTMEATH 2-11 0-10 ANTRIM
    target: Tomas Corrigan shone as Fermanagh beat Laois while Antrim stayed top of Division Four with victory over London.
"""
import os

train_input, train_target = [], []

hardcoded_delete_input = ['Here are our favourites:\n', "Here are some of the Ethiopian runner's greatest feats.\n",
                          'JANUARYFEBRUARYMARCHAPRILMAYJUNE\n',
                          'Donegal 1-14 1-12 MayoDown 0-06 0-22 KerryDerry 2-12 1-18 GalwayLaois 0-14 1-14 TyroneMeath 1-13 1-20 CavanAntrim 2-14 0-09 Leitrim\n',
                          'KERRY 1-13 1-8 DONEGALMONAGHAN 1-12 2-11 MAYOROSCOMMON 1-12 0-6 DOWNFERMANAGH 1-17 0-10 LAOISLONDON 0-11 1-11 ANTRIMAllianz Hurling LeagueWESTMEATH 2-11 0-10 ANTRIM\n']

hardcoded_delete_target = ['On Monday, we asked for you to send us your favourite shop pun names.\n',
                           'Haile Gebrselassie has announced his retirement from competitive running, bringing to an end a 25-year career in which he claimed two Olympic gold medals, eight World Championship victories and set 27 world records.\n',
                           'As 2015 draws to an end, we take a look back at some of the major stories of the year, along with others that proved popular with readers.\n',
                           'FOOTBALL LEAGUE RESULTS\n',
                           'Tomas Corrigan shone as Fermanagh beat Laois while Antrim stayed top of Division Four with victory over London.\n']

with open(f"data/xsum/train.source", "r", encoding='utf8') as f:
    for line in f:
        if line not in hardcoded_delete_input:
            train_input.append(line)

with open(f"data/xsum/train.target", "r", encoding='utf8') as f:
    for line in f:
        if line not in hardcoded_delete_target:
            train_target.append(line)

print(f"there are {len(train_input)} in the new source file")
print(f"there are {len(train_target)} in the new target file")

if os.path.exists("data/xsum/train.source"):
    os.remove("data/xsum/train.source")

if os.path.exists("data/xsum/train.target"):
    os.remove("data/xsum/train.target")

with open(f"data/xsum/train.source", "w", encoding='utf8') as f:
    for item in train_input:
        f.write(item)

with open(f"data/xsum/train.target", "w", encoding='utf8') as f:
    for item in train_target:
        f.write(item)
