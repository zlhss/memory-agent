"""Locked official unseen evaluation for the frozen Memory V2 pipeline."""

import copy
import io
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import main
import memory_search
from memory_v2 import process_user_message_v2
from ollama_client import get_ollama_timings
from regression_test_support import is_environment_error


PROJECT_DIR = Path(__file__).resolve().parent
MEMORY_FILE = PROJECT_DIR / "memories.json"
RESULTS_FILE = PROJECT_DIR / "final_memory_v2_results.json"


FINAL_V2_CASES = [
    {
        "case_id": "final_v2_01_household_bill_responsibility",
        "category": "ADD stable fact",
        "sender_id": "final_v2_target_01",
        "initial_target_memories": ["联系人收藏着一套老式公交票"],
        "sentinel_sender_id": "final_v2_sentinel_01",
        "sentinel_sender_memories": [
            "联系人负责核对工作室的水费账单",
            "联系人喜欢收集地铁线路图",
        ],
        "working_memory": [],
        "user_text": "家里的水电账单一直由我每月核对一次。",
        "expected_behavior": "ADD one stable household responsibility",
        "expected_final_memory_semantics": [
            "preserve the bus-ticket collection",
            "store monthly checking of household utility bills",
        ],
        "protected_unrelated_memories": ["联系人收藏着一套老式公交票"],
        "forbidden_semantics": [],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人收藏着一套老式公交票"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "monthly household utility review",
                    "alternatives": [
                        ["水电账单", "每月", "核对"],
                        ["水电", "每个月", "核对"],
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_02_skincare_preference_constraints",
        "category": "ADD constrained preference",
        "sender_id": "final_v2_target_02",
        "initial_target_memories": ["联系人周末会给自行车做基础保养"],
        "sentinel_sender_id": "final_v2_sentinel_02",
        "sentinel_sender_memories": [
            "联系人偏好带香味的护肤品",
            "联系人每天骑公路车上班",
        ],
        "working_memory": [],
        "user_text": "买护肤品时我只选无香型，而且坚决避开做动物实验的品牌。",
        "expected_behavior": "ADD one preference while preserving both explicit restrictions",
        "expected_final_memory_semantics": [
            "only fragrance-free skincare",
            "avoid brands that conduct animal testing",
            "preserve bicycle maintenance",
        ],
        "protected_unrelated_memories": ["联系人周末会给自行车做基础保养"],
        "forbidden_semantics": [],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人周末会给自行车做基础保养"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "complete constrained skincare preference",
                    "alternatives": [
                        ["护肤品", "无香", "动物实验", "避开"],
                        ["护肤品", "无香型", "动物实验", "不"],
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_03_radio_certificate_deadline",
        "category": "ADD deadline goal",
        "sender_id": "final_v2_target_03",
        "initial_target_memories": ["联系人习惯在雨天听爵士唱片"],
        "sentinel_sender_id": "final_v2_sentinel_03",
        "sentinel_sender_memories": [
            "联系人拥有业余无线电操作证",
            "联系人喜欢听古典唱片",
        ],
        "working_memory": [],
        "user_text": "我把拿到业余无线电操作证当成今年开始准备的重点，希望后年六月前完成。",
        "expected_behavior": "ADD a long-term certification goal with its deadline",
        "expected_final_memory_semantics": [
            "amateur-radio operator certificate",
            "preparation starts this year",
            "deadline before June of the year after next",
        ],
        "protected_unrelated_memories": ["联系人习惯在雨天听爵士唱片"],
        "forbidden_semantics": [],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人习惯在雨天听爵士唱片"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "radio certificate goal and deadline",
                    "alternatives": [
                        ["业余无线电", "操作证", "后年六月"],
                        ["无线电", "证", "后年", "六月"],
                    ],
                },
                {
                    "label": "preparation starts this year",
                    "alternatives": [["今年", "准备"]],
                },
            ],
        },
    },
    {
        "case_id": "final_v2_04_accessibility_manual_project",
        "category": "ADD timed project",
        "sender_id": "final_v2_target_04",
        "initial_target_memories": ["联系人每季度整理一次家庭药箱"],
        "sentinel_sender_id": "final_v2_sentinel_04",
        "sentinel_sender_memories": [
            "联系人参与过无障碍设施调研",
            "联系人每月整理一次工具箱",
        ],
        "working_memory": [],
        "user_text": "接下来四个月，我会每周实地走访一次，为小区编一份无障碍设施手册。",
        "expected_behavior": "ADD one ongoing project with duration and visit frequency",
        "expected_final_memory_semantics": [
            "community accessibility manual project",
            "four-month duration",
            "one field visit per week",
        ],
        "protected_unrelated_memories": ["联系人每季度整理一次家庭药箱"],
        "forbidden_semantics": [],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人每季度整理一次家庭药箱"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "fully constrained accessibility project",
                    "alternatives": [
                        ["四个月", "每周", "一次", "无障碍", "手册"]
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_05_monthly_backup_duplicate",
        "category": "Semantic duplicate",
        "sender_id": "final_v2_target_05",
        "initial_target_memories": [
            "联系人每逢月底都会备份个人电脑的重要文件",
            "联系人喜欢使用纸质日历",
        ],
        "sentinel_sender_id": "final_v2_sentinel_05",
        "sentinel_sender_memories": [
            "联系人每周备份公司电脑",
            "联系人只使用电子日历",
        ],
        "working_memory": [],
        "user_text": "每个月快结束的时候，我都会把电脑里的重要资料做一次备份。",
        "expected_behavior": "Recognize a semantic duplicate and perform no write",
        "expected_final_memory_semantics": ["the initial memory set remains exact"],
        "protected_unrelated_memories": [
            "联系人每逢月底都会备份个人电脑的重要文件",
            "联系人喜欢使用纸质日历",
        ],
        "forbidden_semantics": [],
        "expected_mutation": False,
        "assertions": {
            "exact_set": True,
            "memory_count": 2,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_v2_06_distinct_collection_object",
        "category": "Non-duplicate same category",
        "sender_id": "final_v2_target_06",
        "initial_target_memories": [
            "联系人长期收藏老式机械铅笔",
            "联系人每周给叔叔打一次电话",
        ],
        "sentinel_sender_id": "final_v2_sentinel_06",
        "sentinel_sender_memories": [
            "联系人收藏各地独立书店印章",
            "联系人喜欢自动铅笔",
        ],
        "working_memory": [],
        "user_text": "机械铅笔之外，我还长期收集各地独立书店的印章。",
        "expected_behavior": "ADD a distinct collection without replacing the existing one",
        "expected_final_memory_semantics": [
            "preserve mechanical-pencil collection",
            "add independent-bookstore stamp collection",
        ],
        "protected_unrelated_memories": [
            "联系人长期收藏老式机械铅笔",
            "联系人每周给叔叔打一次电话",
        ],
        "forbidden_semantics": [],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 3,
            "exact_present": [
                "联系人长期收藏老式机械铅笔",
                "联系人每周给叔叔打一次电话",
            ],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "bookstore stamp collection",
                    "alternatives": [
                        ["收集", "独立书店", "印章"],
                        ["收藏", "书店", "印章"],
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_07_shared_to_solo_rental",
        "category": "UPDATE current status",
        "sender_id": "final_v2_target_07",
        "initial_target_memories": [
            "联系人目前租住在一套合租公寓",
            "联系人每季度给祖母寄一次照片",
        ],
        "sentinel_sender_id": "final_v2_sentinel_07",
        "sentinel_sender_memories": [
            "联系人目前租住在一套合租公寓",
            "联系人每月拜访一次祖母",
        ],
        "working_memory": [],
        "user_text": "合租合同到这个月底就结束，之后我已经签好一间独居小公寓。",
        "expected_behavior": "Replace the current shared-rental status with solo renting",
        "expected_final_memory_semantics": [
            "the shared-rental state is gone",
            "solo apartment after the end of this month",
            "the family photo habit remains",
        ],
        "protected_unrelated_memories": ["联系人每季度给祖母寄一次照片"],
        "forbidden_semantics": [
            {
                "label": "old shared rental remains current",
                "alternatives": [["合租公寓"], ["继续合租"]],
            }
        ],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人每季度给祖母寄一次照片"],
            "exact_absent": ["联系人目前租住在一套合租公寓"],
            "required_semantics": [
                {
                    "label": "solo apartment state",
                    "alternatives": [
                        ["独居", "公寓"],
                        ["独自", "公寓"],
                        ["一个人", "公寓"],
                    ],
                },
                {
                    "label": "rental transition timing",
                    "alternatives": [["这个月底"], ["月底", "之后"]],
                },
            ],
        },
    },
    {
        "case_id": "final_v2_08_weekday_sleep_time_update",
        "category": "UPDATE time state",
        "sender_id": "final_v2_target_08",
        "initial_target_memories": [
            "联系人通常晚上十一点半睡觉",
            "联系人每月清洗一次空调滤网",
        ],
        "sentinel_sender_id": "final_v2_sentinel_08",
        "sentinel_sender_memories": [
            "联系人通常晚上十一点半睡觉",
            "联系人习惯早上六点起床",
        ],
        "working_memory": [],
        "user_text": "作息重新调了：现在工作日十点四十五分就睡，不再拖到十一点半。",
        "expected_behavior": "UPDATE the weekday sleep time and remove the old current time",
        "expected_final_memory_semantics": [
            "weekday sleep at 10:45 PM",
            "old 11:30 PM state removed",
            "air-filter maintenance preserved",
        ],
        "protected_unrelated_memories": ["联系人每月清洗一次空调滤网"],
        "forbidden_semantics": [
            {
                "label": "old sleep time remains current",
                "alternatives": [["十一点半", "睡"], ["23:30", "睡"]],
            }
        ],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人每月清洗一次空调滤网"],
            "exact_absent": ["联系人通常晚上十一点半睡觉"],
            "required_semantics": [
                {
                    "label": "new weekday sleep time",
                    "alternatives": [
                        ["工作日", "十点四十五", "睡"],
                        ["工作日", "22:45", "睡"],
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_09_end_herb_garden_care",
        "category": "DELETE explicit termination",
        "sender_id": "final_v2_target_09",
        "initial_target_memories": [
            "联系人每周维护一次阳台香草盆栽",
            "联系人习惯在周五整理电子邮箱",
        ],
        "sentinel_sender_id": "final_v2_sentinel_09",
        "sentinel_sender_memories": [
            "联系人每周维护一次阳台香草盆栽",
            "联系人种植室内蕨类植物",
        ],
        "working_memory": [],
        "user_text": "阳台那批香草我全送人了，以后也不再养，日常养护可以结束了。",
        "expected_behavior": "DELETE the ended herb-care habit only",
        "expected_final_memory_semantics": ["only the email habit remains"],
        "protected_unrelated_memories": ["联系人习惯在周五整理电子邮箱"],
        "forbidden_semantics": [
            {
                "label": "active herb care",
                "alternatives": [["维护", "香草"], ["养护", "香草"]],
            }
        ],
        "expected_mutation": True,
        "expected_exact_memories": ["联系人习惯在周五整理电子邮箱"],
        "assertions": {
            "exact_set": True,
            "memory_count": 1,
            "exact_present": [],
            "exact_absent": ["联系人每周维护一次阳台香草盆栽"],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_v2_10_return_rented_lighting_kit",
        "category": "DELETE paraphrase",
        "sender_id": "final_v2_target_10",
        "initial_target_memories": [
            "联系人租用了按月续费的摄影灯光套装",
            "联系人喜欢冲洗黑白胶卷",
        ],
        "sentinel_sender_id": "final_v2_sentinel_10",
        "sentinel_sender_memories": [
            "联系人租用了按月续费的摄影灯光套装",
            "联系人拥有一台胶片扫描仪",
        ],
        "working_memory": [],
        "user_text": "棚拍那套灯月底就归还，自动续租也关了，后面不会再留着。",
        "expected_behavior": "DELETE the rental using a natural paraphrase",
        "expected_final_memory_semantics": ["film-development preference remains"],
        "protected_unrelated_memories": ["联系人喜欢冲洗黑白胶卷"],
        "forbidden_semantics": [
            {
                "label": "active lighting rental",
                "alternatives": [["租用", "灯光"], ["续费", "摄影灯"]],
            }
        ],
        "expected_mutation": True,
        "expected_exact_memories": ["联系人喜欢冲洗黑白胶卷"],
        "assertions": {
            "exact_set": True,
            "memory_count": 1,
            "exact_present": [],
            "exact_absent": ["联系人租用了按月续费的摄影灯光套装"],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_v2_11_terrace_watering_with_takeout",
        "category": "Persistent plus transient",
        "sender_id": "final_v2_target_11",
        "initial_target_memories": ["联系人喜欢修复旧收音机"],
        "sentinel_sender_id": "final_v2_sentinel_11",
        "sentinel_sender_memories": [
            "联系人每天给露台植物浇水",
            "联系人经常自己做晚饭",
        ],
        "working_memory": [],
        "user_text": "我从这个月开始每天早上给露台植物浇水，眼下正在楼下拿外卖。",
        "expected_behavior": "ADD the lasting watering habit and ignore the current errand",
        "expected_final_memory_semantics": [
            "daily morning terrace watering from this month",
            "no takeaway-pickup memory",
        ],
        "protected_unrelated_memories": ["联系人喜欢修复旧收音机"],
        "forbidden_semantics": [
            {
                "label": "takeaway transient",
                "alternatives": [["外卖"], ["楼下", "拿"]],
            }
        ],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人喜欢修复旧收音机"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "constrained watering habit",
                    "alternatives": [
                        ["这个月", "每天", "早上", "露台", "浇水"]
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_12_family_delivery_and_night_patrol",
        "category": "Two independent ADDs",
        "sender_id": "final_v2_target_12",
        "initial_target_memories": ["联系人使用左手写字"],
        "sentinel_sender_id": "final_v2_sentinel_12",
        "sentinel_sender_memories": [
            "联系人每周给外婆送菜",
            "联系人参加社区白天巡逻",
        ],
        "working_memory": [],
        "user_text": "今年开始我固定参加社区夜间巡逻。还有，每周日我负责给外婆送菜。",
        "expected_behavior": "ADD two separate lasting facts",
        "expected_final_memory_semantics": [
            "community night patrol beginning this year",
            "Sunday grocery delivery for grandmother",
            "facts remain in separate memories",
        ],
        "protected_unrelated_memories": ["联系人使用左手写字"],
        "forbidden_semantics": [],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 3,
            "exact_present": ["联系人使用左手写字"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "night patrol",
                    "alternatives": [["今年", "社区", "夜间巡逻"]],
                },
                {
                    "label": "Sunday grandmother delivery",
                    "alternatives": [["每周日", "外婆", "送菜"]],
                },
            ],
            "distinct_semantic_labels": [
                "night patrol",
                "Sunday grandmother delivery",
            ],
        },
    },
    {
        "case_id": "final_v2_13_bicycle_commute_and_insurance",
        "category": "UPDATE plus ADD",
        "sender_id": "final_v2_target_13",
        "initial_target_memories": [
            "联系人目前乘地铁通勤",
            "联系人每年参加一次家族聚会",
        ],
        "sentinel_sender_id": "final_v2_sentinel_13",
        "sentinel_sender_memories": [
            "联系人目前乘地铁通勤",
            "联系人购买了自行车保险",
        ],
        "working_memory": [],
        "user_text": "刚买了年度自行车保险。通勤这事也变了——从下周起不坐地铁，改骑折叠自行车。",
        "expected_behavior": "UPDATE commute and independently ADD annual insurance",
        "expected_final_memory_semantics": [
            "folding-bike commute from next week",
            "annual bicycle insurance",
            "family gathering remains",
        ],
        "protected_unrelated_memories": ["联系人每年参加一次家族聚会"],
        "forbidden_semantics": [
            {
                "label": "old subway commute",
                "alternatives": [["乘地铁", "通勤"], ["坐地铁", "通勤"]],
            }
        ],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 3,
            "exact_present": ["联系人每年参加一次家族聚会"],
            "exact_absent": ["联系人目前乘地铁通勤"],
            "required_semantics": [
                {
                    "label": "new folding-bike commute",
                    "alternatives": [["下周", "折叠自行车", "通勤"]],
                },
                {
                    "label": "annual bicycle insurance",
                    "alternatives": [["年度", "自行车保险"]],
                },
            ],
            "distinct_semantic_labels": [
                "new folding-bike commute",
                "annual bicycle insurance",
            ],
        },
    },
    {
        "case_id": "final_v2_14_end_newsletter_start_sound_archive",
        "category": "DELETE plus ADD",
        "sender_id": "final_v2_target_14",
        "initial_target_memories": [
            "联系人正在维护一份付费邮件通讯",
            "联系人喜欢收集手绘地图",
        ],
        "sentinel_sender_id": "final_v2_sentinel_14",
        "sentinel_sender_memories": [
            "联系人正在维护一份付费邮件通讯",
            "联系人制作过城市声音作品",
        ],
        "working_memory": [],
        "user_text": "付费通讯到此为止，我不会再更新；接下来会长期制作一套城市声音档案。",
        "expected_behavior": "DELETE the newsletter and ADD the sound-archive project",
        "expected_final_memory_semantics": [
            "paid newsletter no longer active",
            "long-term city sound archive",
            "map collection remains",
        ],
        "protected_unrelated_memories": ["联系人喜欢收集手绘地图"],
        "forbidden_semantics": [
            {
                "label": "active paid newsletter",
                "alternatives": [["维护", "付费邮件通讯"], ["继续", "付费通讯"]],
            }
        ],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人喜欢收集手绘地图"],
            "exact_absent": ["联系人正在维护一份付费邮件通讯"],
            "required_semantics": [
                {
                    "label": "long-term city sound archive",
                    "alternatives": [["长期", "城市声音", "档案"]],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_15_band_exit_subtitle_service_and_rain",
        "category": "Change plus ADD plus transient",
        "sender_id": "final_v2_target_15",
        "initial_target_memories": [
            "联系人每周二参加业余乐队排练",
            "联系人收藏老电影海报",
        ],
        "sentinel_sender_id": "final_v2_sentinel_15",
        "sentinel_sender_memories": [
            "联系人每周二参加业余乐队排练",
            "联系人偶尔为电影制作字幕",
        ],
        "working_memory": [],
        "user_text": "刚才还在雨里等车。对了，乐队排练我已经退出；从下月起每周六为社区无障碍电影放映制作字幕，至少坚持一年。",
        "expected_behavior": "DELETE the rehearsal, ADD the constrained subtitle service, ignore waiting in rain",
        "expected_final_memory_semantics": [
            "old band rehearsal removed",
            "weekly Saturday accessibility subtitles from next month for at least a year",
            "movie-poster collection preserved",
        ],
        "protected_unrelated_memories": ["联系人收藏老电影海报"],
        "forbidden_semantics": [
            {
                "label": "active band rehearsal",
                "alternatives": [["参加", "乐队排练"]],
            },
            {
                "label": "rain waiting transient",
                "alternatives": [["雨里", "等车"], ["正在", "等车"]],
            },
        ],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人收藏老电影海报"],
            "exact_absent": ["联系人每周二参加业余乐队排练"],
            "required_semantics": [
                {
                    "label": "fully constrained subtitle service",
                    "alternatives": [
                        ["下月", "每周六", "无障碍电影", "字幕", "一年"]
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_16_unique_insurance_checklist_reference",
        "category": "Working Memory unique reference",
        "sender_id": "final_v2_target_16",
        "initial_target_memories": [
            "联系人正在整理家庭保险文件清单",
            "联系人习惯在月底检查烟雾报警器",
        ],
        "sentinel_sender_id": "final_v2_sentinel_16",
        "sentinel_sender_memories": [
            "联系人正在整理家庭保险文件清单",
            "联系人准备明年更新保险",
        ],
        "working_memory": [
            {
                "role": "user",
                "content": "家庭保险文件清单还缺两份保单编号。",
            },
            {
                "role": "assistant",
                "content": "那份清单目前还差两项信息。",
            },
        ],
        "user_text": "那份清单我决定延后到明年一月再完成。",
        "expected_behavior": "Resolve the unique checklist referent and UPDATE its schedule",
        "expected_final_memory_semantics": [
            "family insurance checklist completion moved to next January",
            "smoke-alarm habit preserved",
        ],
        "protected_unrelated_memories": ["联系人习惯在月底检查烟雾报警器"],
        "forbidden_semantics": [
            {
                "label": "old unscheduled active checklist",
                "alternatives": [["正在整理", "家庭保险文件清单"]],
            }
        ],
        "expected_mutation": True,
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人习惯在月底检查烟雾报警器"],
            "exact_absent": ["联系人正在整理家庭保险文件清单"],
            "required_semantics": [
                {
                    "label": "rescheduled insurance checklist",
                    "alternatives": [
                        ["家庭保险", "清单", "明年一月"],
                        ["保险文件", "清单", "明年", "一月"],
                    ],
                }
            ],
        },
    },
    {
        "case_id": "final_v2_17_ambiguous_household_documents",
        "category": "Working Memory ambiguous reference",
        "sender_id": "final_v2_target_17",
        "initial_target_memories": [
            "联系人正在编写租客维修指南",
            "联系人正在制作家庭应急物资表",
            "联系人喜欢听现场录音专辑",
        ],
        "sentinel_sender_id": "final_v2_sentinel_17",
        "sentinel_sender_memories": [
            "联系人正在编写租客维修指南",
            "联系人正在制作家庭应急物资表",
        ],
        "working_memory": [
            {
                "role": "user",
                "content": "租客维修指南和家庭应急物资表最近都需要补内容。",
            },
            {
                "role": "assistant",
                "content": "两份资料同时补充确实会占用不少时间。",
            },
        ],
        "user_text": "那项先暂停吧，眼下实在顾不过来。",
        "expected_behavior": "AMBIGUOUS and no write because two referents are plausible",
        "expected_final_memory_semantics": ["the complete initial set remains exact"],
        "protected_unrelated_memories": [
            "联系人正在编写租客维修指南",
            "联系人正在制作家庭应急物资表",
            "联系人喜欢听现场录音专辑",
        ],
        "forbidden_semantics": [],
        "expected_mutation": False,
        "assertions": {
            "exact_set": True,
            "memory_count": 3,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_v2_18_bread_crumb_knowledge_question",
        "category": "No-op knowledge question",
        "sender_id": "final_v2_target_18",
        "initial_target_memories": [
            "联系人习惯周六清洗鱼缸",
            "联系人喜欢阅读建筑杂志",
        ],
        "sentinel_sender_id": "final_v2_sentinel_18",
        "sentinel_sender_memories": [
            "联系人学习过面包发酵",
            "联系人养了两条金鱼",
        ],
        "working_memory": [],
        "user_text": "为什么有些面包烤好后内部会出现很大的孔洞？",
        "expected_behavior": "No write for a world-knowledge question",
        "expected_final_memory_semantics": ["the complete initial set remains exact"],
        "protected_unrelated_memories": [
            "联系人习惯周六清洗鱼缸",
            "联系人喜欢阅读建筑杂志",
        ],
        "forbidden_semantics": [
            {
                "label": "question converted into contact memory",
                "alternatives": [["询问", "面包"], ["想知道", "孔洞"]],
            },
            {
                "label": "bread proposition stored",
                "alternatives": [["面包", "孔洞"]],
            },
        ],
        "expected_mutation": False,
        "assertions": {
            "exact_set": True,
            "memory_count": 2,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_v2_19_sort_camping_list_request",
        "category": "No-op one-time request",
        "sender_id": "final_v2_target_19",
        "initial_target_memories": [
            "联系人每年冬天更换一次净水器滤芯",
            "联系人偏爱亚麻材质的床品",
        ],
        "sentinel_sender_id": "final_v2_sentinel_19",
        "sentinel_sender_memories": [
            "联系人经常整理露营装备",
            "联系人喜欢轻量帐篷",
        ],
        "working_memory": [],
        "user_text": "帮我把这张露营装备清单按重量从轻到重排一下。",
        "expected_behavior": "No write for a one-time sorting request",
        "expected_final_memory_semantics": ["the complete initial set remains exact"],
        "protected_unrelated_memories": [
            "联系人每年冬天更换一次净水器滤芯",
            "联系人偏爱亚麻材质的床品",
        ],
        "forbidden_semantics": [
            {
                "label": "one-time request stored",
                "alternatives": [["露营装备", "清单"], ["按重量", "排序"]],
            }
        ],
        "expected_mutation": False,
        "assertions": {
            "exact_set": True,
            "memory_count": 2,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_v2_20_cross_sender_family_service",
        "category": "Sender isolation",
        "sender_id": "final_v2_target_20",
        "initial_target_memories": [
            "联系人每月为父亲预约一次上门理发服务",
            "联系人喜欢收集电影首映海报",
        ],
        "sentinel_sender_id": "final_v2_sentinel_20",
        "sentinel_sender_memories": [
            "联系人每月为父亲预约一次上门理发服务",
            "联系人喜欢收集舞台剧海报",
        ],
        "working_memory": [],
        "user_text": "父亲已经搬到护理院，那项上门理发预约我取消了。",
        "expected_behavior": "DELETE only the target sender's family service",
        "expected_final_memory_semantics": [
            "target no longer has the haircut appointment",
            "target movie-poster collection remains",
            "sentinel remains byte-for-byte unchanged",
        ],
        "protected_unrelated_memories": ["联系人喜欢收集电影首映海报"],
        "forbidden_semantics": [
            {
                "label": "target haircut service remains active",
                "alternatives": [["预约", "上门理发"]],
            }
        ],
        "expected_mutation": True,
        "expected_exact_memories": ["联系人喜欢收集电影首映海报"],
        "assertions": {
            "exact_set": True,
            "memory_count": 1,
            "exact_present": [],
            "exact_absent": ["联系人每月为父亲预约一次上门理发服务"],
            "required_semantics": [],
        },
    },
]


def _semantic_match(memory, alternatives):
    return any(
        all(term in memory for term in terms)
        for terms in alternatives
    )


def _matching_indices(memories, alternatives):
    return {
        index
        for index, memory in enumerate(memories)
        if _semantic_match(memory, alternatives)
    }


def _has_distinct_assignment(label_indices, labels, position=0, used=None):
    used = used or set()
    if position == len(labels):
        return True
    for index in label_indices.get(labels[position], set()):
        if index not in used and _has_distinct_assignment(
            label_indices,
            labels,
            position + 1,
            used | {index},
        ):
            return True
    return False


def check_expected_state(case, final_memories):
    assertions = case["assertions"]
    errors = []

    if assertions.get("exact_set"):
        expected = sorted(
            case.get("expected_exact_memories", case["initial_target_memories"])
        )
        if sorted(final_memories) != expected:
            errors.append("final memory set does not match exactly")

    expected_count = assertions.get("memory_count")
    if expected_count is not None and len(final_memories) != expected_count:
        errors.append(
            f"memory count expected {expected_count}, got {len(final_memories)}"
        )

    for memory in assertions.get("exact_present", []):
        if memory not in final_memories:
            errors.append(f"required exact memory missing: {memory}")

    for memory in assertions.get("exact_absent", []):
        if memory in final_memories:
            errors.append(f"forbidden exact memory still present: {memory}")

    label_indices = {}
    for semantic in assertions.get("required_semantics", []):
        indices = _matching_indices(final_memories, semantic["alternatives"])
        label_indices[semantic["label"]] = indices
        if not indices:
            errors.append(f"required semantics missing: {semantic['label']}")

    distinct_labels = assertions.get("distinct_semantic_labels", [])
    if distinct_labels and not _has_distinct_assignment(
        label_indices,
        distinct_labels,
    ):
        errors.append(
            "independent facts were not preserved in distinct memories: "
            + ", ".join(distinct_labels)
        )

    for semantic in case["forbidden_semantics"]:
        if _matching_indices(final_memories, semantic["alternatives"]):
            errors.append(f"forbidden semantics present: {semantic['label']}")

    return errors


def destructive_errors(case, final_memories, sentinel_unchanged):
    errors = []
    for memory in case["protected_unrelated_memories"]:
        if memory not in final_memories:
            errors.append(f"protected unrelated memory changed or removed: {memory}")
    if not sentinel_unchanged:
        errors.append("sentinel sender memories changed")
    return errors


def reset_case(case):
    memory_search.reset_memories({
        case["sender_id"]: copy.deepcopy(case["initial_target_memories"]),
        case["sentinel_sender_id"]: copy.deepcopy(
            case["sentinel_sender_memories"]
        ),
    })
    main.reset_working_memories()
    working_memory = main.get_working_memory(case["sender_id"])
    for message in case["working_memory"]:
        working_memory.add_message(message["role"], message["content"])


def evaluate_case(case):
    reset_case(case)
    output = io.StringIO()
    started = time.perf_counter()
    response = None

    try:
        with redirect_stdout(output):
            response = process_user_message_v2(
                case["sender_id"],
                case["user_text"],
            )
        elapsed = time.perf_counter() - started
        final_memories = memory_search.get_sender_memories(
            case["sender_id"]
        ).copy()
        sentinel_memories = memory_search.get_sender_memories(
            case["sentinel_sender_id"]
        ).copy()
        sentinel_unchanged = (
            sentinel_memories == case["sentinel_sender_memories"]
        )
        behavior_errors = check_expected_state(case, final_memories)
        destructive = destructive_errors(
            case,
            final_memories,
            sentinel_unchanged,
        )
        if not behavior_errors and not destructive:
            status = "PASS"
        elif destructive:
            status = "DESTRUCTIVE_FAILURE"
        else:
            status = "BEHAVIOR_FAIL"

        profile = response.get("timing_profile", {})
        processed_actions = response.get("processed_actions", [])
        safe_abstain = (
            status == "BEHAVIOR_FAIL"
            and case["expected_mutation"]
            and not processed_actions
        )
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "status": status,
            "elapsed_seconds": elapsed,
            "llm_calls": profile.get("total_llm_calls", 0),
            "cross_sender_pollution": not sentinel_unchanged,
            "safe_abstain": safe_abstain,
            "behavior_errors": behavior_errors,
            "destructive_errors": destructive,
            "initial_memories": case["initial_target_memories"],
            "working_memory": case["working_memory"],
            "atomic_facts": response.get("atomic_facts", []),
            "per_fact": [
                {
                    "atomic_fact": item.get("atomic_fact"),
                    "retrieval_candidates": (
                        item.get("retrieval") or {}
                    ).get("candidates", []),
                    "alignment": item.get("alignment"),
                    "validation": item.get("validation"),
                    "action": item.get("action"),
                    "applied": item.get("applied"),
                }
                for item in response.get("fact_results", [])
            ],
            "final_memories": final_memories,
            "sentinel_memories": sentinel_memories,
            "expected_semantics": case["expected_final_memory_semantics"],
            "captured_log": output.getvalue(),
        }
    except Exception as error:
        elapsed = time.perf_counter() - started
        calls = get_ollama_timings()
        sentinel_memories = memory_search.get_sender_memories(
            case["sentinel_sender_id"]
        ).copy()
        sentinel_unchanged = (
            sentinel_memories == case["sentinel_sender_memories"]
        )
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "status": (
                "ENVIRONMENT_ERROR"
                if is_environment_error(error)
                else "PROGRAM_ERROR"
            ),
            "elapsed_seconds": elapsed,
            "llm_calls": len(calls),
            "cross_sender_pollution": not sentinel_unchanged,
            "safe_abstain": False,
            "behavior_errors": [],
            "destructive_errors": (
                [] if sentinel_unchanged else ["sentinel sender memories changed"]
            ),
            "initial_memories": case["initial_target_memories"],
            "working_memory": case["working_memory"],
            "atomic_facts": (
                response.get("atomic_facts", [])
                if isinstance(response, dict)
                else []
            ),
            "per_fact": (
                response.get("fact_results", [])
                if isinstance(response, dict)
                else []
            ),
            "final_memories": memory_search.get_sender_memories(
                case["sender_id"]
            ).copy(),
            "sentinel_memories": sentinel_memories,
            "expected_semantics": case["expected_final_memory_semantics"],
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "captured_log": output.getvalue(),
        }


def validate_benchmark():
    errors = []
    required_fields = {
        "case_id",
        "category",
        "sender_id",
        "initial_target_memories",
        "sentinel_sender_id",
        "sentinel_sender_memories",
        "working_memory",
        "user_text",
        "expected_behavior",
        "expected_final_memory_semantics",
        "protected_unrelated_memories",
        "forbidden_semantics",
        "expected_mutation",
        "assertions",
    }
    if len(FINAL_V2_CASES) != 20:
        errors.append(f"case count is {len(FINAL_V2_CASES)}, expected 20")
    case_ids = [case["case_id"] for case in FINAL_V2_CASES]
    if len(set(case_ids)) != len(case_ids):
        errors.append("case_id values are not unique")
    for case in FINAL_V2_CASES:
        missing = required_fields - set(case)
        if missing:
            errors.append(f"{case.get('case_id')}: missing fields {sorted(missing)}")
        if not case.get("sentinel_sender_memories"):
            errors.append(f"{case.get('case_id')}: missing sentinel memories")
        if case.get("sender_id") == case.get("sentinel_sender_id"):
            errors.append(f"{case.get('case_id')}: sender and sentinel IDs collide")
    return errors


def print_specs():
    print("FINAL MEMORY V2 OFFICIAL BENCHMARK SPECIFICATIONS")
    for index, case in enumerate(FINAL_V2_CASES, start=1):
        print(
            f"{index:02d}. {case['case_id']} | {case['category']} | "
            f"{case['user_text']} -> {case['expected_behavior']}"
        )


def run_official_evaluation():
    results = []
    for index, case in enumerate(FINAL_V2_CASES, start=1):
        print(
            f"[{index:02d}/20] Running {case['case_id']}...",
            flush=True,
        )
        result = evaluate_case(case)
        results.append(result)
        print(
            f"{case['case_id']} | {case['category']} | {result['status']} | "
            f"{result['elapsed_seconds']:.3f}s | "
            f"llm_calls={result['llm_calls']} | "
            f"cross_sender_pollution="
            f"{'YES' if result['cross_sender_pollution'] else 'NO'}",
            flush=True,
        )
        if result["status"] != "PASS":
            failure_details = {
                key: result[key]
                for key in (
                    "initial_memories",
                    "working_memory",
                    "atomic_facts",
                    "per_fact",
                    "final_memories",
                    "sentinel_memories",
                    "expected_semantics",
                    "behavior_errors",
                    "destructive_errors",
                    "safe_abstain",
                )
            }
            if result.get("error"):
                failure_details["error"] = result["error"]
                failure_details["traceback"] = result["traceback"]
            print(
                "FAILURE_DETAILS:",
                json.dumps(failure_details, ensure_ascii=False, indent=2),
                flush=True,
            )
    return results


def print_summary(results):
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in (
            "PASS",
            "BEHAVIOR_FAIL",
            "DESTRUCTIVE_FAILURE",
            "PROGRAM_ERROR",
            "ENVIRONMENT_ERROR",
        )
    }
    total = len(results)
    average_latency = (
        sum(result["elapsed_seconds"] for result in results) / total
        if total
        else 0.0
    )
    average_calls = (
        sum(result["llm_calls"] for result in results) / total
        if total
        else 0.0
    )
    safe_abstains = sum(result["safe_abstain"] for result in results)
    pollution_count = sum(
        result["cross_sender_pollution"] for result in results
    )
    category_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    for result in results:
        stats = category_stats[result["category"]]
        stats["total"] += 1
        stats["passed"] += result["status"] == "PASS"

    print("\nFINAL MEMORY V2 OFFICIAL SUMMARY")
    print(f"PASS: {counts['PASS']}/{total}")
    print(f"FINAL_PASS_RATE: {counts['PASS'] / total:.2%}")
    print(f"BEHAVIOR_FAIL: {counts['BEHAVIOR_FAIL']}")
    print(f"DESTRUCTIVE_FAILURE: {counts['DESTRUCTIVE_FAILURE']}")
    print(f"SAFE_ABSTAIN_FAILURE: {safe_abstains}")
    print(f"PROGRAM_ERROR: {counts['PROGRAM_ERROR']}")
    print(f"ENVIRONMENT_ERROR: {counts['ENVIRONMENT_ERROR']}")
    print(f"CROSS_SENDER_POLLUTION: {pollution_count}")
    print(f"AVERAGE_LATENCY_SECONDS: {average_latency:.3f}")
    print(f"AVERAGE_LLM_CALLS: {average_calls:.2f}")
    print("BY_CATEGORY:")
    for category, stats in category_stats.items():
        print(f"  {category}: {stats['passed']}/{stats['total']}")


def main_entry():
    validation_errors = validate_benchmark()
    if validation_errors:
        for error in validation_errors:
            print(f"VALIDATION_ERROR: {error}")
        return 2

    if "--validate" in sys.argv:
        print("BENCHMARK_VALIDATION: PASS")
        return 0
    if "--specs" in sys.argv:
        print_specs()
        return 0

    original_memories = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    results = None
    try:
        results = run_official_evaluation()
        print_summary(results)
        RESULTS_FILE.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0
    finally:
        memory_search.reset_memories(original_memories)
        main.reset_working_memories()
        print("Original memories restored; Working Memory cleared.")


if __name__ == "__main__":
    raise SystemExit(main_entry())
