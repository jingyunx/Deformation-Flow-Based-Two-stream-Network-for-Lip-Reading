import os
import numpy as np
import glob
import time
import cv2
from torch.utils.data import Dataset
from .cvtransforms import *
import torch
import pickle
import getpass
import random 
random.seed()

def load_file(filename):
    arrays = np.load(filename)
    arrays = arrays / 255.
    return arrays

class LRWDataset(Dataset):
    def build_file_list(self, dir, set):
        labels = ['SENIOR', 'WORLD', 'IMMIGRATION', 'OPERATION', 'LEVEL', 'REPORT', 'PROTECT', 'OFFICIALS', 'YOUNG', 'MAJOR', 'SINCE', 'DIFFICULT', 'ANSWER', 'RESULT', 'SINGLE', 'HAPPENED', 'HAPPENING', 'MAYBE', 'DOING', 'MINISTERS', 'ISLAMIC', 'JUSTICE', 'LEAVE', 'REMEMBER', 'PLACES', 'CHARGE', 'MATTER', 'PARTY', 'LATER', 'PARTIES', 'HAPPEN', 'VIOLENCE', 'WANTED', 'PLACE', 'NEVER', 'COUNTRIES', 'LATEST', 'RUSSIAN', 'EVENTS', 'RUSSIA', 'FRIDAY', 'MINUTES', 'FIGURES', 'POWER', 'MILITARY', 'OPPOSITION', 'RATES', 'WITHOUT', 'WOULD', 'FOLLOWING', 'AGREEMENT', 'VICTIMS', 'PHONE', 'ACTION', 'LEADERSHIP', 'TERMS', 'BUILDING', 'HISTORY', 'SPEECH', 'HAVING', 'SOUTH', 'EXTRA', 'POSITION', 'PARLIAMENT', 'STREET', 'BRITISH', 'STAND', 'MONTHS', 'HEAVY', 'BANKS', 'EVERYBODY', 'POTENTIAL', 'CHINA', 'SENSE', 'UNION', 'OTHER', 'PRESIDENT', 'WHICH', 'WELCOME', 'WEEKS', 'PRESS', 'CONCERNS', 'MANCHESTER', 'COURT', 'DETAILS', 'CAMERON', 'FRENCH', 'MEANS', 'THINGS', 'HIGHER', 'SCHOOLS', 'CENTRAL', 'PRIME', 'ORDER', 'PROBLEMS', 'HOUSING', 'CALLED', 'SYRIAN', 'AGREE', 'ACROSS', 'GOING', 'COMMUNITY', 'SMALL', 'PAYING', 'PRETTY', 'WINDS', 'PATIENTS', 'GROUP', 'SAYING', 'PUBLIC', 'ARRESTED', 'DAVID', 'MEASURES', 'NOTHING', 'PARENTS', 'WESTERN', 'RUNNING', 'TAKEN', 'COMPANY', 'ACCESS', 'STILL', 'PERSON', 'HUNDREDS', 'SOUTHERN', 'POLITICIANS', 'SECRETARY', 'LONDON', 'LABOUR', 'MEMBERS', 'INSIDE', 'CHIEF', 'LEAST', 'ALWAYS', 'ACCUSED', 'RIGHTS', 'JAMES', 'PLANS', 'CLOSE', 'WATCHING', 'MINISTER', 'BETWEEN', 'ALLOW', 'COMING', 'ASKED', 'WELFARE', 'WOMEN', 'PRESSURE', 'FAMILIES', 'BENEFITS', 'ITSELF', 'TOMORROW', 'CASES', 'SCOTTISH', 'MARKET', 'FORCE', 'TEMPERATURES', 'THEIR', 'EDUCATION', 'ALLEGATIONS', 'TRUST', 'INVESTMENT', 'STATEMENT', 'TIMES', 'PRICES', 'REALLY', 'CHILDREN', 'OFFICE', 'RIGHT', 'THROUGH', 'SOMETHING', 'POSSIBLE', 'SPENT', 'GROUND', 'SERVICES', 'GAMES', 'EXPECT', 'COUPLE', 'CRIME', 'CERTAINLY', 'INCREASE', 'MEDIA', 'COMPANIES', 'CUSTOMERS', 'COMES', 'AMOUNT', 'WANTS', 'POINT', 'CONTROL', 'DECISION', 'HOMES', 'MEDICAL', 'LEADERS', 'CONFERENCE', 'KILLED', 'GREECE', 'EXAMPLE', 'SUNSHINE', 'SOCIETY', 'SEEMS', 'MEMBER', 'GETTING', 'SPENDING', 'FOCUS', 'CONSERVATIVE', 'SUNDAY', 'PRICE', 'REFERENDUM', 'INVOLVED', 'TAKING', 'CHALLENGE', 'TRIAL', 'ANOTHER', 'WATER', 'WEAPONS', 'WHETHER', 'CONFLICT', 'AMERICAN', 'GROWING', 'BORDER', 'SOMEONE', 'BEHIND', 'CURRENT', 'PROBABLY', 'EVIDENCE', 'GUILTY', 'UNDERSTAND', 'CANNOT', 'SIGNIFICANT', 'COUNCIL', 'SIMPLY', 'GIVING', 'PERIOD', 'WORST', 'HOUSE', 'LEVELS', 'THREE', 'EVENING', 'AFFAIRS', 'MIDDLE', 'THERE', 'ECONOMIC', 'BLACK', 'SITUATION', 'COURSE', 'ATTACKS', 'AHEAD', 'LARGE', 'CLEAR', 'ABOUT', 'ALLOWED', 'INDUSTRY', 'TALKS', 'SEVEN', 'VOTERS', 'LIVES', 'SYRIA', 'FURTHER', 'WHILE', 'AFRICA', 'MAKES', 'LOCAL', 'SPEAKING', 'ABUSE', 'PARTS', 'CAPITAL', 'NATIONAL', 'WARNING', 'DECIDED', 'WORDS', 'DEBATE', 'PROCESS', 'EUROPE', 'PROBLEM', 'FAMILY', 'ATTACK', 'THING', 'RESPONSE', 'MASSIVE', 'MISSING', 'AFFECTED', 'MOVING', 'BENEFIT', 'RULES', 'CAMPAIGN', 'STRONG', 'PRISON', 'YESTERDAY', 'BECAUSE', 'EVERYTHING', 'GIVEN', 'COUNTRY', 'YEARS', 'QUITE', 'NEEDS', 'INTEREST', 'LONGER', 'DEFICIT', 'PROVIDE', 'CHANGE', 'POLICE', 'DESPITE', 'LOOKING', 'MURDER', 'TRADE', 'FINAL', 'LEADER', 'QUESTIONS', 'AREAS', 'DESCRIBED', 'FACING', 'STATES', 'SHOULD', 'SCOTLAND', 'AGAINST', 'FOOTBALL', 'CHANCE', 'THREAT', 'SOCIAL', 'EXPECTED', 'AUTHORITIES', 'RECENT', 'RECORD', 'GERMANY', 'INDEPENDENT', 'ECONOMY', 'DEATH', 'MORNING', 'FUTURE', 'ALREADY', 'BEFORE', 'PEOPLE', 'BEING', 'EASTERN', 'FORMER', 'WEATHER', 'HEALTH', 'SECTOR', 'FRANCE', 'SECURITY', 'NUMBER', 'EVERYONE', 'THEMSELVES', 'OFFICERS', 'EUROPEAN', 'BELIEVE', 'BRITAIN', 'PERHAPS', 'IMPORTANT', 'HEART', 'MEETING', 'WRONG', 'CHILD', 'WAITING', 'INQUIRY', 'CRISIS', 'LITTLE', 'DIFFERENT', 'ELECTION', 'BROUGHT', 'EDITOR', 'GREAT', 'MILLIONS', 'TODAY', 'SERIOUS', 'ENGLAND', 'CHARGES', 'SERIES', 'FRONT', 'FORWARD', 'TOGETHER', 'BECOME', 'TALKING', 'FIRST', 'LIKELY', 'POLITICAL', 'HUMAN', 'DEGREES', 'ASKING', 'FIGHT', 'SPECIAL', 'HEARD', 'WESTMINSTER', 'OTHERS', 'BETTER', 'BIGGEST', 'AFTERNOON', 'CHANGES', 'WALES', 'RETURN', 'ISSUE', 'SERVICE', 'ANYTHING', 'SYSTEM', 'WHOLE', 'ENERGY', 'BILLION', 'BUDGET', 'MONEY', 'AROUND', 'AFTER', 'INFLATION', 'WHERE', 'CONTINUE', 'HOURS', 'NORTH', 'FINANCIAL', 'CANCER', 'THINK', 'USING', 'SIDES', 'TOWARDS', 'FOREIGN', 'NUMBERS', 'PRIVATE', 'DURING', 'FORCES', 'REPORTS', 'OBAMA', 'EVERY', 'EMERGENCY', 'THIRD', 'STAGE', 'BUSINESS', 'OUTSIDE', 'BRING', 'UNTIL', 'THOSE', 'ANNOUNCED', 'UNDER', 'GEORGE', 'FIGHTING', 'STORY', 'MILLION', 'MIGHT', 'IMPACT', 'POWERS', 'ABSOLUTELY', 'POLITICS', 'EXACTLY', 'WEEKEND', 'BUILD', 'ACCORDING', 'FOUND', 'GROWTH', 'GENERAL', 'THOUSANDS', 'WORKERS', 'INFORMATION', 'UNITED', 'REASON', 'MESSAGE', 'MAKING', 'MAJORITY', 'HOSPITAL', 'THOUGHT', 'GOVERNMENT', 'TRYING', 'CLAIMS', 'WITHIN', 'WORKING', 'SECOND', 'KNOWN', 'ACTUALLY', 'SUPPORT', 'LIVING', 'STARTED', 'GLOBAL', 'PERSONAL', 'RATHER', 'AMONG', 'QUESTION', 'OFTEN', 'EARLY', 'NIGHT', 'STATE', 'SCHOOL', 'DIFFERENCE', 'START', 'JUDGE', 'ENOUGH', 'ISSUES', 'MONTH', 'BUSINESSES', 'CLOUD', 'LEGAL', 'AMERICA', 'NORTHERN', 'SPEND', 'SHORT', 'TONIGHT', 'SEVERAL', 'THESE', 'STAFF', 'COULD', 'POLICY', 'ALMOST', 'AGAIN', 'IRELAND', 'MOMENT', 'MIGRANTS']
        completeList = []
        for i, label in enumerate(labels):
            dirpath = dir + "/{}/{}".format(label, set)
            print(i, label, dirpath)
            files = os.listdir(dirpath)
            for file in files:
                if(self.datatype == 'image'):
                    filepath = dirpath + "/{}".format(file)
                    entry = (i, filepath)
                    completeList.append(entry)
                if(self.datatype == 'video'):
                    if file.endswith("mp4"):
                        filepath = dirpath + "/{}".format(file)
                        entry = (i, filepath)
                        completeList.append(entry)    
        return  completeList

    def __init__(self, path, set):
        self.set = set 
        self.file_list = self.build_file_list(directory, set)
        print('Total num of samples: ', len(self.file_list))

    def __getitem__(self, idx):
        path = self.file_list[idx][1]  
        inputs = load_file(path)
        if(self.set == 'train'):
            batch_img = RandomCrop(inputs, (88, 88))
        batch_img = inputs
        label = self.file_list[idx][0]
        vid_tensor =  torch.FloatTensor(batch_img[:, np.newaxis,...])
        sample =  {'x': vid_tensor, 'label': torch.LongTensor([int(label)])}
        return sample 

    def __len__(self):
        return len(self.file_list)