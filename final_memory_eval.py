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

import main
import memory_search
from ollama_client import get_ollama_timings
from regression_test_support import is_environment_error


PROJECT_DIR = Path(__file__).resolve().parent
MEMORY_FILE = PROJECT_DIR / "memories.json"


FINAL_CASES = [
    {
        "case_id": "final_01_pet_identity",
        "category": "ADD stable fact",
        "sender_id": "final_target_01",
        "initial_memories": [
            "联系人每月参观一次博物馆",
        ],
        "working_memory": [],
        "user_message": "我家里养着一只六岁的橘猫，名字叫年糕。",
        "expected_final_memories": [
            "联系人每月参观一次博物馆",
            "联系人养着一只六岁、名叫年糕的橘猫",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_01",
        "sentinel_sender_memories": [
            "联系人养着一只叫可乐的边牧",
            "联系人住在海边",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人每月参观一次博物馆"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "pet identity",
                    "alternatives": [["年糕", "猫"], ["年糕", "橘猫"]],
                },
                {
                    "label": "pet age",
                    "alternatives": [["六岁"], ["6岁"]],
                },
            ],
        },
    },
    {
        "case_id": "final_02_hotpot_preference",
        "category": "ADD preference",
        "sender_id": "final_target_02",
        "initial_memories": [
            "联系人在一家物流公司工作",
        ],
        "working_memory": [],
        "user_message": "吃火锅我只点清汤锅，辣的我完全受不了。",
        "expected_final_memories": [
            "联系人吃火锅偏好清汤锅并且不能吃辣",
            "联系人在一家物流公司工作",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_02",
        "sentinel_sender_memories": [
            "联系人特别喜欢麻辣火锅",
            "联系人经营一家面包店",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人在一家物流公司工作"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "clear broth preference",
                    "alternatives": [["火锅", "清汤"]],
                },
                {
                    "label": "spicy aversion",
                    "alternatives": [
                        ["辣", "受不了"],
                        ["辣", "不能"],
                        ["不吃辣"],
                    ],
                },
            ],
        },
    },
    {
        "case_id": "final_03_marathon_goal",
        "category": "ADD long-term goal",
        "sender_id": "final_target_03",
        "initial_memories": [
            "联系人养了一缸热带鱼",
        ],
        "working_memory": [],
        "user_message": "我给自己定了目标，明年年底前要完成第一次全程马拉松。",
        "expected_final_memories": [
            "联系人养了一缸热带鱼",
            "联系人计划在明年年底前完成第一次全程马拉松",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_03",
        "sentinel_sender_memories": [
            "联系人每年参加一次半程马拉松",
            "联系人喜欢拼装模型",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人养了一缸热带鱼"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "full marathon goal",
                    "alternatives": [["全程马拉松"], ["全马"]],
                },
                {
                    "label": "goal deadline",
                    "alternatives": [["明年", "年底"], ["明年底"]],
                },
            ],
        },
    },
    {
        "case_id": "final_04_botanical_illustration_project",
        "category": "ADD ongoing project",
        "sender_id": "final_target_04",
        "initial_memories": [
            "联系人习惯早晨听新闻",
        ],
        "working_memory": [],
        "user_message": "我正在做一套城市植物插画集，接下来半年会持续更新。",
        "expected_final_memories": [
            "联系人习惯早晨听新闻",
            "联系人正在创作城市植物插画集，并计划持续更新半年",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_04",
        "sentinel_sender_memories": [
            "联系人正在制作海洋摄影集",
            "联系人喜欢收集邮票",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人习惯早晨听新闻"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "creative project",
                    "alternatives": [
                        ["城市植物", "插画"],
                        ["植物", "插画集"],
                    ],
                },
                {
                    "label": "project duration",
                    "alternatives": [["半年"], ["六个月"]],
                },
            ],
        },
    },
    {
        "case_id": "final_05_bedtime_reading_duplicate",
        "category": "Semantic duplicate",
        "sender_id": "final_target_05",
        "initial_memories": [
            "联系人每天睡前阅读半小时历史类书籍",
            "联系人喜欢喝柠檬水",
        ],
        "working_memory": [],
        "user_message": "我晚上临睡前都会看一会儿历史书，差不多半个钟头。",
        "expected_final_memories": [
            "联系人每天睡前阅读半小时历史类书籍",
            "联系人喜欢喝柠檬水",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_05",
        "sentinel_sender_memories": [
            "联系人每天清晨读财经新闻",
            "联系人不喝含糖饮料",
        ],
        "assertions": {
            "exact_set": True,
            "memory_count": 2,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_06_distinct_beverage_preference",
        "category": "Non-duplicate same category",
        "sender_id": "final_target_06",
        "initial_memories": [
            "联系人喜欢喝手冲咖啡",
            "联系人对花生过敏",
        ],
        "working_memory": [],
        "user_message": "除了咖啡，我也很喜欢喝焙火乌龙茶。",
        "expected_final_memories": [
            "联系人喜欢喝手冲咖啡",
            "联系人对花生过敏",
            "联系人也喜欢喝焙火乌龙茶",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_06",
        "sentinel_sender_memories": [
            "联系人不喜欢喝茶",
            "联系人对乳制品过敏",
        ],
        "assertions": {
            "memory_count": 3,
            "exact_present": [
                "联系人喜欢喝手冲咖啡",
                "联系人对花生过敏",
            ],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "oolong tea preference",
                    "alternatives": [["喜欢", "乌龙茶"], ["偏好", "乌龙茶"]],
                },
            ],
        },
    },
    {
        "case_id": "final_07_fulltime_to_freelance",
        "category": "UPDATE current status",
        "sender_id": "final_target_07",
        "initial_memories": [
            "联系人目前在一家广告公司做全职文案",
            "联系人养了一只金毛犬",
        ],
        "working_memory": [],
        "user_message": "这个月起我已经转成自由职业了，主要接品牌文案项目。",
        "expected_final_memories": [
            "联系人目前是自由职业者，主要承接品牌文案项目",
            "联系人养了一只金毛犬",
        ],
        "forbidden_memories": [
            "联系人目前在一家广告公司做全职文案",
        ],
        "forbidden_concepts": [
            {
                "label": "old full-time status",
                "alternatives": [["广告公司", "全职"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_07",
        "sentinel_sender_memories": [
            "联系人目前在一家广告公司做全职文案",
            "联系人喜欢养多肉植物",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人养了一只金毛犬"],
            "exact_absent": ["联系人目前在一家广告公司做全职文案"],
            "required_semantics": [
                {
                    "label": "freelance work",
                    "alternatives": [
                        ["自由职业", "文案"],
                        ["自由职业者", "品牌"],
                    ],
                },
            ],
        },
    },
    {
        "case_id": "final_08_podcast_release_cadence",
        "category": "UPDATE numeric status",
        "sender_id": "final_target_08",
        "initial_memories": [
            "联系人运营的播客每月更新一期",
            "联系人喜欢徒步",
        ],
        "working_memory": [],
        "user_message": "播客的排期改了，往后每两周更新一期。",
        "expected_final_memories": [
            "联系人运营的播客每两周更新一期",
            "联系人喜欢徒步",
        ],
        "forbidden_memories": [
            "联系人运营的播客每月更新一期",
        ],
        "forbidden_concepts": [
            {
                "label": "old monthly cadence",
                "alternatives": [["播客", "每月"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_08",
        "sentinel_sender_memories": [
            "联系人运营的播客每月更新一期",
            "联系人喜欢露营",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人喜欢徒步"],
            "exact_absent": ["联系人运营的播客每月更新一期"],
            "required_semantics": [
                {
                    "label": "fortnightly cadence",
                    "alternatives": [
                        ["播客", "每两周"],
                        ["播客", "两周", "一期"],
                    ],
                },
            ],
        },
    },
    {
        "case_id": "final_09_close_bakery_stall",
        "category": "DELETE explicit termination",
        "sender_id": "final_target_09",
        "initial_memories": [
            "联系人正在经营周末烘焙摊位",
            "联系人喜欢观测星空",
        ],
        "working_memory": [],
        "user_message": "周末市集的烘焙摊我已经彻底收掉了，以后不摆了。",
        "expected_final_memories": [
            "联系人喜欢观测星空",
        ],
        "forbidden_memories": [
            "联系人正在经营周末烘焙摊位",
        ],
        "forbidden_concepts": [
            {
                "label": "active bakery stall",
                "alternatives": [["正在经营", "烘焙摊"], ["继续", "摆摊"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_09",
        "sentinel_sender_memories": [
            "联系人正在经营周末烘焙摊位",
            "联系人喜欢日出摄影",
        ],
        "assertions": {
            "exact_set": True,
            "memory_count": 1,
            "exact_present": [],
            "exact_absent": ["联系人正在经营周末烘焙摊位"],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_10_cancel_flower_subscription",
        "category": "DELETE paraphrase",
        "sender_id": "final_target_10",
        "initial_memories": [
            "联系人订阅了每周五送达的鲜花包月服务",
            "联系人住在江边",
        ],
        "working_memory": [],
        "user_message": "以后周五不用再等花了，我把那个按月配送的服务退掉了。",
        "expected_final_memories": [
            "联系人住在江边",
        ],
        "forbidden_memories": [
            "联系人订阅了每周五送达的鲜花包月服务",
        ],
        "forbidden_concepts": [
            {
                "label": "active flower subscription",
                "alternatives": [["鲜花", "订阅"], ["鲜花", "包月"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_10",
        "sentinel_sender_memories": [
            "联系人订阅了每周五送达的鲜花包月服务",
            "联系人住在山脚",
        ],
        "assertions": {
            "exact_set": True,
            "memory_count": 1,
            "exact_present": [],
            "exact_absent": ["联系人订阅了每周五送达的鲜花包月服务"],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_11_breakfast_habit_with_transient",
        "category": "Persistent plus transient",
        "sender_id": "final_target_11",
        "initial_memories": [
            "联系人周末喜欢逛旧书店",
        ],
        "working_memory": [],
        "user_message": "以后早餐我都喝无糖豆浆，刚刚还把钥匙忘在办公室了。",
        "expected_final_memories": [
            "联系人周末喜欢逛旧书店",
            "联系人以后早餐固定喝无糖豆浆",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [
            {
                "label": "forgotten key transient",
                "alternatives": [["钥匙"], ["忘在办公室"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_11",
        "sentinel_sender_memories": [
            "联系人早餐经常喝牛奶",
            "联系人在科技园工作",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人周末喜欢逛旧书店"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "soy milk breakfast habit",
                    "alternatives": [["早餐", "无糖豆浆"]],
                },
            ],
        },
    },
    {
        "case_id": "final_12_volunteer_and_adoption",
        "category": "Two independent ADDs",
        "sender_id": "final_target_12",
        "initial_memories": [
            "联系人住在合肥",
        ],
        "working_memory": [],
        "user_message": "下个月起我周末会去做动物救助志愿者，还领养了一只叫豆包的三花猫。",
        "expected_final_memories": [
            "联系人住在合肥",
            "联系人下个月起周末参加动物救助志愿服务",
            "联系人领养了一只叫豆包的三花猫",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_12",
        "sentinel_sender_memories": [
            "联系人在儿童图书馆做志愿者",
            "联系人养了一只叫布丁的狸花猫",
        ],
        "assertions": {
            "memory_count": 3,
            "exact_present": ["联系人住在合肥"],
            "exact_absent": [],
            "required_semantics": [
                {
                    "label": "animal rescue volunteering",
                    "alternatives": [["动物救助", "志愿"]],
                },
                {
                    "label": "adopted cat",
                    "alternatives": [["豆包", "猫"], ["豆包", "三花"]],
                },
            ],
        },
    },
    {
        "case_id": "final_13_vehicle_update_and_first_aid_course",
        "category": "UPDATE plus ADD",
        "sender_id": "final_target_13",
        "initial_memories": [
            "联系人目前开燃油车通勤",
            "联系人喜欢看自然纪录片",
        ],
        "working_memory": [],
        "user_message": "我把油车卖了，通勤换成纯电车；另外还报了周末的急救证书课程。",
        "expected_final_memories": [
            "联系人目前开纯电车通勤",
            "联系人喜欢看自然纪录片",
            "联系人报名了周末急救证书课程",
        ],
        "forbidden_memories": [
            "联系人目前开燃油车通勤",
        ],
        "forbidden_concepts": [
            {
                "label": "old combustion commute",
                "alternatives": [["燃油车", "通勤"], ["油车", "通勤"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_13",
        "sentinel_sender_memories": [
            "联系人目前开燃油车通勤",
            "联系人正在学道路救援",
        ],
        "assertions": {
            "memory_count": 3,
            "exact_present": ["联系人喜欢看自然纪录片"],
            "exact_absent": ["联系人目前开燃油车通勤"],
            "required_semantics": [
                {
                    "label": "electric vehicle commute",
                    "alternatives": [["纯电车", "通勤"], ["电动车", "通勤"]],
                },
                {
                    "label": "first aid course",
                    "alternatives": [["急救", "课程"], ["急救", "证书"]],
                },
            ],
        },
    },
    {
        "case_id": "final_14_close_blog_start_podcast",
        "category": "DELETE plus ADD",
        "sender_id": "final_target_14",
        "initial_memories": [
            "联系人正在维护一个个人美食博客",
            "联系人喜欢乘坐绿皮火车旅行",
        ],
        "working_memory": [],
        "user_message": "美食博客我决定关掉，接下来会长期经营一档人物访谈播客。",
        "expected_final_memories": [
            "联系人喜欢乘坐绿皮火车旅行",
            "联系人接下来会长期经营一档人物访谈播客",
        ],
        "forbidden_memories": [
            "联系人正在维护一个个人美食博客",
        ],
        "forbidden_concepts": [
            {
                "label": "active food blog",
                "alternatives": [["正在", "美食博客"], ["维护", "美食博客"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_14",
        "sentinel_sender_memories": [
            "联系人正在维护一个个人美食博客",
            "联系人喜欢听人物访谈播客",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人喜欢乘坐绿皮火车旅行"],
            "exact_absent": ["联系人正在维护一个个人美食博客"],
            "required_semantics": [
                {
                    "label": "new interview podcast",
                    "alternatives": [["人物访谈", "播客"], ["访谈播客"]],
                },
            ],
        },
    },
    {
        "case_id": "final_15_choir_history_and_low_battery",
        "category": "Change plus ADD plus transient",
        "sender_id": "final_target_15",
        "initial_memories": [
            "联系人每周参加社区合唱团排练",
            "联系人喜欢吃黑巧克力",
        ],
        "working_memory": [],
        "user_message": "我已经退出社区合唱团，同时决定用一年写完一本家族口述史，刚才手机只剩百分之二的电。",
        "expected_final_memories": [
            "联系人喜欢吃黑巧克力",
            "联系人计划用一年写完一本家族口述史",
        ],
        "forbidden_memories": [
            "联系人每周参加社区合唱团排练",
        ],
        "forbidden_concepts": [
            {
                "label": "active choir participation",
                "alternatives": [["参加", "合唱团"], ["合唱团", "排练"]],
            },
            {
                "label": "low battery transient",
                "alternatives": [["手机", "电"], ["百分之二"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_15",
        "sentinel_sender_memories": [
            "联系人每周参加社区合唱团排练",
            "联系人正在整理家谱",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人喜欢吃黑巧克力"],
            "exact_absent": ["联系人每周参加社区合唱团排练"],
            "required_semantics": [
                {
                    "label": "oral history writing goal",
                    "alternatives": [
                        ["一年", "家族口述史"],
                        ["家族", "口述史", "写"],
                    ],
                },
            ],
        },
    },
    {
        "case_id": "final_16_working_memory_trip_reference",
        "category": "Working Memory reference",
        "sender_id": "final_target_16",
        "initial_memories": [
            "联系人正在筹备今年秋天的冰岛自驾旅行",
            "联系人喜欢制作旅行手账",
        ],
        "working_memory": [
            {
                "role": "user",
                "content": "冰岛路线的住宿我还没有订完。",
            },
            {
                "role": "assistant",
                "content": "这趟自驾还有几处住宿需要确认。",
            },
        ],
        "user_message": "那个我想改到明年春天再去。",
        "expected_final_memories": [
            "联系人计划明年春天进行冰岛自驾旅行",
            "联系人喜欢制作旅行手账",
        ],
        "forbidden_memories": [
            "联系人正在筹备今年秋天的冰岛自驾旅行",
        ],
        "forbidden_concepts": [
            {
                "label": "old autumn schedule",
                "alternatives": [["冰岛", "今年秋天"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_16",
        "sentinel_sender_memories": [
            "联系人正在筹备今年秋天的冰岛自驾旅行",
            "联系人喜欢收集登机牌",
        ],
        "assertions": {
            "memory_count": 2,
            "exact_present": ["联系人喜欢制作旅行手账"],
            "exact_absent": ["联系人正在筹备今年秋天的冰岛自驾旅行"],
            "required_semantics": [
                {
                    "label": "rescheduled Iceland trip",
                    "alternatives": [["冰岛", "明年春天"], ["冰岛", "明年", "春"]],
                },
            ],
        },
    },
    {
        "case_id": "final_17_ambiguous_working_memory_reference",
        "category": "Working Memory ambiguity",
        "sender_id": "final_target_17",
        "initial_memories": [
            "联系人正在改造家里的厨房",
            "联系人正在整理祖父留下的旧照片",
            "联系人喜欢听古典音乐",
        ],
        "working_memory": [
            {
                "role": "user",
                "content": "厨房改造和旧照片整理最近都挺费时间。",
            },
            {
                "role": "assistant",
                "content": "这两件事同时推进确实会很累。",
            },
        ],
        "user_message": "这个我先放一放。",
        "expected_final_memories": [
            "联系人正在改造家里的厨房",
            "联系人正在整理祖父留下的旧照片",
            "联系人喜欢听古典音乐",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [],
        "sentinel_sender_id": "final_sentinel_17",
        "sentinel_sender_memories": [
            "联系人正在翻修阳台",
            "联系人正在整理毕业相册",
        ],
        "assertions": {
            "exact_set": True,
            "memory_count": 3,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_18_migration_knowledge_question",
        "category": "No-op knowledge question",
        "sender_id": "final_target_18",
        "initial_memories": [
            "联系人喜欢观察湿地鸟类",
            "联系人住在苏州河附近",
        ],
        "working_memory": [],
        "user_message": "候鸟为什么能辨认迁徙方向？",
        "expected_final_memories": [
            "联系人喜欢观察湿地鸟类",
            "联系人住在苏州河附近",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [
            {
                "label": "question as contact memory",
                "alternatives": [["询问", "候鸟"], ["想知道", "迁徙"]],
            },
            {
                "label": "world knowledge proposition",
                "alternatives": [["候鸟", "辨认", "方向"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_18",
        "sentinel_sender_memories": [
            "联系人研究城市鸟类迁徙",
            "联系人住在湖边",
        ],
        "assertions": {
            "exact_set": True,
            "memory_count": 2,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_19_one_time_meeting_request",
        "category": "No-op one-time request",
        "sender_id": "final_target_19",
        "initial_memories": [
            "联系人每周日给父母打电话",
            "联系人喜欢木质香调",
        ],
        "working_memory": [],
        "user_message": "帮我把这段会议记录整理成三条要点。",
        "expected_final_memories": [
            "联系人每周日给父母打电话",
            "联系人喜欢木质香调",
        ],
        "forbidden_memories": [],
        "forbidden_concepts": [
            {
                "label": "one-time request",
                "alternatives": [["会议记录"], ["整理", "要点"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_19",
        "sentinel_sender_memories": [
            "联系人负责记录部门周会",
            "联系人喜欢柑橘香调",
        ],
        "assertions": {
            "exact_set": True,
            "memory_count": 2,
            "exact_present": [],
            "exact_absent": [],
            "required_semantics": [],
        },
    },
    {
        "case_id": "final_20_identical_cross_sender_memory",
        "category": "Sender isolation",
        "sender_id": "final_target_20",
        "initial_memories": [
            "联系人正在照顾一只叫奶糖的兔子",
            "联系人周末在社区图书馆兼职",
        ],
        "working_memory": [],
        "user_message": "奶糖已经送给我姐姐长期照顾了，我以后不养它了。",
        "expected_final_memories": [
            "联系人周末在社区图书馆兼职",
        ],
        "forbidden_memories": [
            "联系人正在照顾一只叫奶糖的兔子",
        ],
        "forbidden_concepts": [
            {
                "label": "target still caring for rabbit",
                "alternatives": [["正在照顾", "奶糖"]],
            },
        ],
        "sentinel_sender_id": "final_sentinel_20",
        "sentinel_sender_memories": [
            "联系人正在照顾一只叫奶糖的兔子",
            "联系人喜欢拼图",
        ],
        "assertions": {
            "exact_set": True,
            "memory_count": 1,
            "exact_present": [],
            "exact_absent": ["联系人正在照顾一只叫奶糖的兔子"],
            "required_semantics": [],
        },
    },
]


def _semantic_match(memory, alternatives):
    return any(
        all(keyword in memory for keyword in keywords)
        for keywords in alternatives
    )


def _semantic_exists(memories, alternatives):
    return any(
        _semantic_match(memory, alternatives)
        for memory in memories
    )


def _check_expected_state(case, final_memories):
    assertions = case["assertions"]
    errors = []

    if assertions.get("exact_set"):
        if sorted(final_memories) != sorted(
            case["expected_final_memories"]
        ):
            errors.append("final memory set does not match exactly")

    expected_count = assertions.get("memory_count")
    if expected_count is not None and len(final_memories) != expected_count:
        errors.append(
            f"memory count expected {expected_count}, "
            f"got {len(final_memories)}"
        )

    for memory in assertions.get("exact_present", []):
        if memory not in final_memories:
            errors.append(f"required exact memory missing: {memory}")

    for memory in (
        assertions.get("exact_absent", [])
        + case["forbidden_memories"]
    ):
        if memory in final_memories:
            errors.append(f"forbidden exact memory present: {memory}")

    for requirement in assertions.get("required_semantics", []):
        if not _semantic_exists(
            final_memories,
            requirement["alternatives"],
        ):
            errors.append(
                f"required semantic memory missing: "
                f"{requirement['label']}"
            )

    for forbidden in case["forbidden_concepts"]:
        if _semantic_exists(
            final_memories,
            forbidden["alternatives"],
        ):
            errors.append(
                f"forbidden semantic memory present: "
                f"{forbidden['label']}"
            )

    invalid_schema = [
        memory
        for memory in final_memories
        if not main.is_valid_contact_memory(memory)
    ]
    if invalid_schema:
        errors.append(
            f"contact memory schema violation: {invalid_schema}"
        )

    return errors


def _log_values(log, prefix):
    return [
        line[len(prefix):].strip()
        for line in log.splitlines()
        if line.startswith(prefix)
    ]


def _failure_diagnostics(case, log, final_memories):
    return {
        "initial_memories": case["initial_memories"],
        "working_memory": case["working_memory"],
        "current_user_message": case["user_message"],
        "contextual_query": _log_values(log, "Contextual Query:"),
        "relevant_memories": _log_values(log, "Relevant Memories:"),
        "raw_memory_manager_actions": _log_values(log, "Raw Actions:"),
        "coverage_checker_actions": _log_values(
            log,
            "Missing Coverage Actions:",
        ),
        "add_evidence_results": _log_values(log, "ADD Evidence:"),
        "worthiness_results": _log_values(log, "Worthiness:"),
        "duplicate_candidates": _log_values(
            log,
            "Duplicate Candidates:",
        ),
        "duplicate_results": _log_values(log, "Duplicate Check:"),
        "final_memories": final_memories,
        "expected_result": {
            "expected_final_memories": case["expected_final_memories"],
            "forbidden_memories": case["forbidden_memories"],
            "forbidden_concepts": case["forbidden_concepts"],
            "assertions": case["assertions"],
        },
    }


def _reset_case(case):
    memory_search.reset_memories({
        case["sender_id"]: copy.deepcopy(
            case["initial_memories"]
        ),
        case["sentinel_sender_id"]: copy.deepcopy(
            case["sentinel_sender_memories"]
        ),
    })
    main.reset_working_memories()
    working_memory = main.get_working_memory(case["sender_id"])

    for message in case["working_memory"]:
        working_memory.add_message(
            message["role"],
            message["content"],
        )


def evaluate_case(case):
    _reset_case(case)
    output = io.StringIO()
    started = time.perf_counter()
    response_result = None
    error = None
    exception_text = None

    try:
        with redirect_stdout(output):
            response_result = main.process_user_message(
                case["sender_id"],
                case["user_message"],
            )
    except Exception as caught:
        error = caught
        exception_text = traceback.format_exc()

    elapsed = time.perf_counter() - started
    log = output.getvalue()
    llm_timings = get_ollama_timings()
    final_memories = memory_search.get_sender_memories(
        case["sender_id"]
    ).copy()
    sentinel_memories = memory_search.get_sender_memories(
        case["sentinel_sender_id"]
    ).copy()
    sentinel_unchanged = (
        sentinel_memories == case["sentinel_sender_memories"]
    )

    if error is not None:
        status = (
            "ENVIRONMENT_ERROR"
            if is_environment_error(error)
            else "PROGRAM_ERROR"
        )
        errors = [repr(error)]
    else:
        errors = _check_expected_state(case, final_memories)
        if not sentinel_unchanged:
            errors.append("cross-sender pollution detected")
        status = "PASS" if not errors else "BEHAVIOR_FAIL"

    result = {
        "case_id": case["case_id"],
        "category": case["category"],
        "status": status,
        "elapsed_seconds": round(elapsed, 3),
        "llm_calls": len(llm_timings),
        "llm_elapsed_seconds": round(sum(
            timing["elapsed_seconds"]
            for timing in llm_timings
        ), 3),
        "sender_pollution": not sentinel_unchanged,
        "errors": errors,
    }

    if status != "PASS":
        result["diagnostics"] = _failure_diagnostics(
            case,
            log,
            final_memories,
        )
        result["sentinel_memories"] = sentinel_memories
        result["exception"] = exception_text

    return result


def print_specs():
    for index, case in enumerate(FINAL_CASES, start=1):
        print(
            f"{index:02d}. {case['case_id']} | {case['category']}\n"
            f"    sender={case['sender_id']}\n"
            f"    initial={case['initial_memories']}\n"
            f"    working={case['working_memory']}\n"
            f"    message={case['user_message']}\n"
            f"    expected={case['expected_final_memories']}\n"
            f"    forbidden={case['forbidden_memories']} / "
            f"{[item['label'] for item in case['forbidden_concepts']]}\n"
            f"    sentinel={case['sentinel_sender_memories']}"
        )


def print_summary(results):
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in (
            "PASS",
            "BEHAVIOR_FAIL",
            "PROGRAM_ERROR",
            "ENVIRONMENT_ERROR",
        )
    }
    category_results = defaultdict(lambda: {"passed": 0, "total": 0})

    for result in results:
        stats = category_results[result["category"]]
        stats["total"] += 1
        if result["status"] == "PASS":
            stats["passed"] += 1

    total_elapsed = sum(
        result["elapsed_seconds"]
        for result in results
    )
    average_latency = total_elapsed / len(results)
    average_llm_calls = sum(
        result["llm_calls"]
        for result in results
    ) / len(results)
    pollution_count = sum(
        result["sender_pollution"]
        for result in results
    )

    summary = {
        **counts,
        "cross_sender_pollution_count": pollution_count,
        "final_pass_rate": counts["PASS"] / len(results),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "average_latency_seconds": round(average_latency, 3),
        "average_llm_calls": round(average_llm_calls, 3),
        "category_results": dict(category_results),
    }
    print("FINAL_SUMMARY", json.dumps(summary, ensure_ascii=False))


def run_official_evaluation():
    original_memories = json.loads(
        MEMORY_FILE.read_text(encoding="utf-8")
    )
    results = []

    try:
        for index, case in enumerate(FINAL_CASES, start=1):
            print(
                f"[{index:02d}/20] Running {case['case_id']}...",
                flush=True,
            )
            result = evaluate_case(case)
            results.append(result)
            print(
                "FINAL_CASE_RESULT",
                json.dumps(result, ensure_ascii=False),
                flush=True,
            )
    finally:
        memory_search.reset_memories(original_memories)
        main.reset_working_memories()
        print("FINAL_STATE_RESTORED", flush=True)

    print_summary(results)
    return results


if __name__ == "__main__":
    if "--specs" in sys.argv:
        print_specs()
    else:
        run_official_evaluation()
