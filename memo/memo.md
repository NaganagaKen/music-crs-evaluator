- countvectorizerではなく、TF-IDFのほうがいいかも
  - exp007-009あたり、過去コンペの解法を見ていて思ったこと


**exp005でまだ使っていない主なデータ**
1. **Track-Metadata の未使用列**
   `track_name`, `artist_name`, `album_name`, `ISRC`, `popularity`, `release_date`, `duration` は exp005 では未使用です。ローカル Arrow では 47,071 tracks 中、`track_name/artist_name/album_name/popularity/duration` は全件埋まっていて、`release_date` も 46,427 件、`ISRC` も 45,737 件あります。  
   item2vec なら、例えば `__pop_bin__:40-50`, `__year__:2006`, `__decade__:2000s`, `__duration_bin__:3-4min`, `__artist_name__:...` のような属性トークンとして track と共起させられます。

2. **tag_list の残り**
   exp005 は `max_tags_per_track=10` で、各 track の tag を最大10個に切っています。実データでは `tag_list` の平均長が約33.49なので、多くの tag はまだ捨てています。  
   まずは `max_tags_per_track=-1`, `20`, `5` の ablation が素直です。ただし globally frequent tag だけ増やすと汎用語が強くなりすぎるので、低頻度タグ寄り、TF-IDF風、または tag relation の重み下げも候補です。

3. **train 会話の user / assistant / thought**
   train には `user`, `music`, `assistant` がそれぞれ 121,592 行ありますが、今は `music` だけ使っています。  
   item2vec のままなら、各 turn を `[user_text_tokens, goal_tokens, profile_tokens, track_id]` のような共起列にして、自然言語側のトークンと track を同じ Word2Vec 空間に入れられます。これは Union ではなく、item2vec のコーパス拡張です。

4. **user_profile / conversation_goal**
   train/test 両方に `age_group`, `country_code`, `gender`, `preferred_musical_culture`, `conversation_goal.category`, `listener_goal`, `specificity` があります。今の exp005 では未使用です。  
   特に `preferred_musical_culture` と `conversation_goal` は track 選択に効きそうです。dev/test でも取得できるので、query 側にも入れやすいです。

5. **user_id**
   dev/test の 500 user のうち 371 user は train 側にも存在していました。test session 数では `test_warm=800`, `test_cold=200` です。  
   item2vec で `__user__:user_id` を track と共起させると warm user には効く可能性があります。ただし cold には効かないので、profile token と併用するのがよさそうです。

**優先して試すなら**
一番 item2vec らしくて筋がいいのは、exp005 をベースにした **context-augmented item2vec** です。

- track sequence は現状維持
- metadata pair に `popularity_bin`, `release_year/decade`, `duration_bin`, `artist_name`, `album_name` を追加
- train turn ごとに `user_profile`, `conversation_goal`, user 発話キーワードを track_id と共起
- dev query は「過去 track 平均」だけでなく、現在 turn までの profile/goal/user text token のベクトルも平均に混ぜる

注意点として、dev/test の `music` 行は正解 track なので、現在 turn や未来 turn の `music` を読むのはリークです。使うなら過去 turn の music だけ、または profile/goal/user発話など非正解情報に限定するのが安全です。コード変更はしていません。