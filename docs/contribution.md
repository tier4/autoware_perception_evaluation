## coding rule

- See [detail](https://github.com/tier4/AWMLtools/blob/main/docs/development/contribution.md)

## Branch rule
### Branch

- Main branch: logsimに使われているbranch
  - mergeするたびにversionが上がる(ex. 1.0 -> 1.1)
  - 基本的にdevelopブランチ以外をmergeしない
- Develop branch: 開発用branch
  - topic_branchからのpull_requestはここに受ける
- topic_branch
  - developに取り込んだのち削除
  - 開発速度優先で細かくcommitしコミットメッセージもある程度適当でも良い
  - feature/fix などのprefixをつける

### Merge rule

- topic branch-> develop branch
  - Squash and mergeでPRごとのcommitにする
- develop branch -> master branch
  - Create a merge commitでPRごとのcommitを維持する
  - アプリケーションとの結合を行うリリース作業に当たる
  - 何か問題が出たらtopic branch release/v1.x などを作って結合作業を行う

## library の構成について

- ros package化
  - 使われる時はros packageとしてincludeされる
    - autoware_utilsと同様な感じになる
  - package.xmlを追加する
  - 縛り：/awml_evaluation/awml_evaluation 以下に全てのcodeを入れる(ROSパッケージとしてimportするため)
