from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from datetime import datetime, timedelta
from pathlib import Path 
from time import sleep
import re

from pandas import DataFrame

BASE_URL = r"https://holodex.net/{suffix}"
MSEDGE_DRIVER = "C:/Program Files (x86)/Microsoft/Edge/Application/msedgedriver.exe"
MSEDGE_USERDATA = '..\\..\\..\\..\\AppData\\Local\\Microsoft\\Edge\\User Data'

vspo_search = BASE_URL.format(
    suffix="search?q=type,value,text%0Aorg,VSpo,VSpo&page={pagenum}&type=clip")

hdex_settings = BASE_URL.format(suffix="settings")

# -------------------- Open a selenium-controlled browser -------------------- #


def open_edge_driver(
        driver_path=MSEDGE_DRIVER,
        user_data=MSEDGE_USERDATA) -> webdriver:

    options = webdriver.EdgeOptions()
    
    # options.add_argument(f"user-data-dir={user_data}")
    options.add_argument('--allow-running-insecure-content')
    options.add_argument('--ignore-certificate-errors')

    service = Service(driver_path)
    driver = webdriver.Edge(service=service, options=options)
    driver.maximize_window()
    return driver


# -------------------------- Navigate Holodex's page ------------------------- #


def add_jp_clips(driver: webdriver, link=hdex_settings) -> None:
    """add JP clips to settings"""
    driver.get(hdex_settings)
    box = driver.find_element(
        by=By.XPATH,
        value="//*[@id=\"app\"]/div/main/div/div[2]/div/div[2]/div/div[2]/div[5]/div/div[1]/div/div"
    )
    box.click()
    return


class VideoInfoParser:
    def __init__(self, videos: list[WebElement], now: datetime = None) -> None:
        self._videos = videos
        self._now = datetime.now() if not now else now
        self.UPLOAD_TIME_PAT = re.compile(
            r"(\d{1,2}\/\d{1,2}\/\d{4})\s\(([\d\:]{4,5})\s(PM|AM)\)")

    def read_text(self, v: WebElement) -> tuple[str]:
        return v.find_element(By.TAG_NAME, 'a').\
            text.split('\n')

    @staticmethod
    def get_duration(dur: str, secs=(3600, 60, 1)) -> int:
        times = tuple(map(int, dur.split(":")))

        if len(times) < 3:
            return sum(a*b for a, b in zip(secs[1:], times))
        else:
            return sum(a*b for a, b in zip(secs, times))

    def get_uploadTime(self, day_time: str) -> datetime:

        if 'ago' in day_time:
            num, unit, _ = day_time.split(' ')
            return self._now - timedelta(**{unit: int(num)})
        else:
            day, hhmm, ampm = self.UPLOAD_TIME_PAT.\
                search(day_time).\
                groups()
            
            # if res is None:
            #     print(day_time)
            #     raise TypeError(f"No timestamp found.")
            # else:
            #     day, hhmm, ampm = res.groups()

            hm = f"0{hhmm}" if len(hhmm) < 5 else hhmm

            return datetime.strptime(
                f"{day} {hm} {ampm}",
                "%m/%d/%Y %I:%M %p"
            )

    def parse_all(
            self, info: list = None) -> list[tuple[datetime, int, str]]:

        if info is None:
            info: list[tuple[datetime, int, str]] = []

        for video in self._videos:
            duration, _, channel, uploadTime = self.read_text(video)

            info.append(
                (
                    self.get_uploadTime(uploadTime),
                    self.get_duration(duration),
                    channel
                )
            )

        print("Parsed all videos!")
        return info

def open_hdex() -> webdriver:
    driver = open_edge_driver() 
    add_jp_clips(driver)
    return driver 

def search_holodex(
    first_page = 1,
    last_page = 100,
    driver: webdriver=None, 
    data: list[tuple] = None,
    filename: str="./data/vspo_clippers/") -> DataFrame:
    
    data = data if data else [] 
    parser = None

    assert first_page < last_page 
    current_page_num = first_page 

    outfile = filename + "h_page-{p}.part"

    while current_page_num < last_page:
        if driver is None:
            driver = open_hdex()
            sleep(12)

        print(f"Parsing page {current_page_num} of {last_page}...")
        driver.get(vspo_search.format(
            pagenum=current_page_num))

        sleep(5)

        videos = driver.find_elements(
            By.XPATH,
            "//*[@id=\"app\"]/div/main/div/div[2]/div[2]/div/div/div[2]/div/div"
        )

        sleep(2)

        if parser is None:
            parser = VideoInfoParser(videos)
        else:
            parser._videos = videos

        try:
            data = parser.parse_all(data)
        except Exception as e:
            print(f"Early interrupt at page {current_page_num} due to", e, sep='\n')
            break 
        
        sleep(5)

        if current_page_num % 5 == 0:
            DataFrame.from_records(data).\
                to_csv(outfile.format(p=current_page_num))

            print("Saved.")
            data = [] 

        current_page_num += 1

    return data 
    
# ---------------------------------------------------------------------------- #


driver = open_hdex()

data = search_holodex(
    first_page=106, 
    last_page=116,
    driver=driver,
)

driver.close()