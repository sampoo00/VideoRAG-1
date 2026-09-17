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

from videorag._llm import *
from videorag import VideoRAG, QueryParam

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    #query = 'What is the relationship between Iron Man and Spider-Man? How do they meet, and how does Iron Man help Spider-Man?'
    query = '사람들이 어디에 모여있고, 무슨 스포츠를 보고 있나요?'
    param = QueryParam(mode="videorag")
    # if param.wo_reference = False, VideoRAG will add reference to video clips in the response
    param.wo_reference = True

    videorag = videorag = VideoRAG(llm=openai_4o_mini_config, working_dir=f"./videorag-workdir")
    videorag.load_caption_model(debug=False)
    response = videorag.query(query=query, param=param)
    print(response)