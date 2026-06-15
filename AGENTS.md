- 私の許可なくコードを変更するのは**絶対にやめてください**
- これはRecsys Challenge 2026の分析のためのリポジトリです。
  - 必要であればRecsys Challenge 2026のホームページやhugging faceも参照してください。
  - 楽曲情報はリークに当たらないものとして扱い、Track-Metadata と Track-Embeddings は train/test に分けず、利用可能な全楽曲を使用してください。

- retrieval の保存形式は以下を参考にしてください。
[
  {
    "session_id": "ba3da7b0-1e81-4d2a-90fa-65ee1f4d7348",
    "user_id": "32957cc8-8e3e-4903-8f82-d0b917abff52",
    "turn_number": 1,
    "predicted_track_ids": [
      "2445ed62-2e19-4222-8d01-3a57f685755d",
      "a8ff5da0-a9ce-4e97-8ae4-12e7acb5e463",
      "d67f0dce-62aa-4efb-94f5-7ce3dba62ca9",
      "0ebbe1fa-8705-4f68-998b-fca589381bf7",
      "3ffb6f21-6344-4fef-8be2-8f4645cbc56a",
      "be80761d-f178-4fd4-a60d-f9c47d44f758",
      "0037f4ad-071a-4c18-92ac-65841b807bb6",
      "16bccd41-e989-42c5-9a11-8b5582442f75",
       ],
    "predicted_track_scores": [
      0.3119330406188965,
      0.2023497372865677,
      0.0828627198934555,
      0.07593265920877457,
      0.0690968930721283,
      0.06717860698699951,
      0.06639391928911209,
      0.06245192512869835,
      0.06160913407802582,
      0.059590306133031845,
      0.05622313171625137,
      0.05272083356976509,
      0.05263584852218628,
      0.05150983855128288,
      0.02534700743854046
    ],
    "predicted_response": "tfidf cosine retrieval"
  },
  {
    "session_id": "ba3da7b0-1e81-4d2a-90fa-65ee1f4d7348",
    "user_id": "32957cc8-8e3e-4903-8f82-d0b917abff52",
    "turn_number": 2,
    "predicted_track_ids": [
      "2445ed62-2e19-4222-8d01-3a57f685755d",
      "a8ff5da0-a9ce-4e97-8ae4-12e7acb5e463",
      "27275a5c-c0df-4de7-830c-c56525356e05",
      "954bebb2-ffb7-43ba-a6d2-5a71b0d37eb2",
      "7f5f9197-e413-4a12-9267-40aa33fd804e"
    ]
    ...
  }
  ...
]
