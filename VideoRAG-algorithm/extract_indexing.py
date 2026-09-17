import os
import logging
import warnings
import multiprocessing

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from videorag._llm import openai_4o_mini_config
from videorag import VideoRAG, QueryParam


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Please enter your video file path in this list; there is no limit on the length.
    # Here is an example; you can use your own videos instead.
    video_paths = [
        #'movies/Iron-Man.mp4',
        #'movies/Spider-Man.mkv',
        'movies/crowd.mp4',
        # 'movies/crowd1.mp4',
        # 'movies/crowd2.mp4',
        # 'movies/crowd3.mp4',
        # 'movies/crowd4.mp4',
        # 'movies/crowd5.mp4',
        # 'movies/crowd6.mp4',
    ]
    videorag = VideoRAG(llm=openai_4o_mini_config, working_dir=f"./videorag-workdir")
    videorag.insert_video(video_path_list=video_paths)