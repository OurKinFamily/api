#!/usr/bin/env bash
source /home/stephen/Documents/ourkin/api/venv/bin/activate && \
  DEV_USER_EMAIL=stephenyoung7267@gmail.com \
  OURKIN_ENV=dev \
  uvicorn app.main:app --reload
