#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import logging
import warnings
import multiprocessing
import nest_asyncio

import sys
from pathlib import Path

# 현재 파일의 상위 상위 디렉토리(프로젝트 루트)를 sys.path 최우선순위로 등록
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# 가상환경 내부의 nvidia cudnn lib 경로 자동 추가
cudnn_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cudnn", "lib")
if os.path.exists(cudnn_path):
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + cudnn_path


nest_asyncio.apply()

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from videorag._llm import  openai_config, openai_4o_mini_config, azure_openai_config, ollama_config
from videorag import VideoRAG, QueryParam


if __name__ == '__main__':
        # In[ ]:


        video_paths = [
                #'/path/to/your/video.mp4',
                '/mnt/d/agent_projects/video_grounding/VideoRAG/VideoRAG-algorithm/movies/crowd.mp4'
        ]

        # In[ ]:


        # Set start method to 'spawn' if not already set (avoid RuntimeError)
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
        
        # In[ ]:


        videorag = VideoRAG(llm=ollama_config, working_dir=f"./videorag-workdir/crowd")

        # In[ ]:


        # To build
        videorag.insert_video(video_path_list=video_paths)

        # In[ ]:


        # To query
        videorag.load_caption_model(debug=False)
        param = QueryParam(mode="videorag")

        # In[ ]:


        #query = "What are the Lexington school construction options"
        query = "사람들이 어디에 모여있고, 무슨 스포츠를 보고 있나요?"
        param.wo_reference = False
        response = videorag.query(query=query, param=param)
        print(response)

        # In[ ]:



