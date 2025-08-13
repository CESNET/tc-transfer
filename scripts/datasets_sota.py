from enum import Enum


class SotaMetricEnum(Enum):
    ACCURACY = "Accuracy"
    MACRO_F1_SCORE = "Macro F1 Score"
    WEIGHTED_F1_SCORE = "Weighted F1 Score"

    def __str__(self):
        return self.value

DATASETS_SOTA = {
    "ISCXVPN2016-App": (SotaMetricEnum.ACCURACY, 63.92),           # if using payload: 79.92
    "ISCXVPN2016-TrafficType": (SotaMetricEnum.ACCURACY, 65.56),   # if using payload: 81.71
    "ISCXVPN2016-Encapsulation": (SotaMetricEnum.ACCURACY, 85.45), # if using payload: 93.01
    "MIRAGE19": (SotaMetricEnum.WEIGHTED_F1_SCORE, 80.06),
    "MIRAGE22": (SotaMetricEnum.WEIGHTED_F1_SCORE, 97.18),
    "UTMOBILENET21": (SotaMetricEnum.WEIGHTED_F1_SCORE, 81.91),
    "UCDAVIS19-Script": (SotaMetricEnum.ACCURACY, 98.63),
    "UCDAVIS19-Human": (SotaMetricEnum.ACCURACY, 80.45),
    "CESNET-TLS22": (SotaMetricEnum.ACCURACY, 97.2),
    "AppClassNet": (SotaMetricEnum.ACCURACY, 88.3),
}
