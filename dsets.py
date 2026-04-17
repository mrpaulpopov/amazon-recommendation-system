from collections import namedtuple
import glob
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

def getCandidateInfoList(requireOnDisk_bool=True):
    # --- серии, доступные на диске ---
    mhd_list = glob.glob('/data/subset*/*.mhd') # glob позволяет искать файлы по шаблону.
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list} # [-1] берет только имя файла без остального файла, а [:-4] удаляет расширение с точкой. В итоге это список всех скачанных mhd

    # --- загрузка CSV ---
    annotations_df = pd.read_csv('data/annotations.csv')
    candidates_df = pd.read_csv('data/candidates.csv')

    # --- фильтр по наличию на диске ---
    if requireOnDisk_bool:
        candidates_df = candidates_df[
            candidates_df['seriesuid'].isin(presentOnDisk_set)
        ]

    # --- подготовка аннотаций ---
    annotations_grouped = annotations_df.groupby('seriesuid')

    candidateInfo_list = []

    # --- обработка кандидатов ---
    for _, row in candidates_df.iterrows():
        series_uid = row['seriesuid']
        candidate_center = (row['coordX'], row['coordY'], row['coordZ'])
        is_nodule = bool(row['class'])

        candidate_diameter = 0.0

        if series_uid in annotations_grouped.groups:
            ann_df = annotations_grouped.get_group(series_uid)

            # векторное сравнение
            deltas = (
                (ann_df[['coordX', 'coordY', 'coordZ']].values - candidate_center)
                .__abs__()
            )

            diameters = ann_df['diameter_mm'].values

            mask = (deltas <= (diameters[:, None] / 4)).all(axis=1)

            if mask.any():
                candidate_diameter = diameters[mask][0]

        candidateInfo_list.append(
            CandidateInfoTuple(
                is_nodule,
                candidate_diameter,
                series_uid,
                candidate_center,
            )
        )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        ct_a.clip(-1000, 1000, ct_a)
        self.series_uid = series_uid
        self.hu_a = ct_a