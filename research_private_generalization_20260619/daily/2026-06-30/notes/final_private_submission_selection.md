# 최종 private 제출 선택

최종 private/generalization 제출 파일은 다음으로 확정했습니다.

- `outputs/FINAL_PRIVATE_CV_OOF970659.csv`
- OOF balanced accuracy: 0.9706589608
- 원본 후보: `outputs/193_PRIVATE_CV_oof970659.csv`
- SHA-256: `fb02a1248ed7cda8d0eebe9fa3a88d37a752f619fe713bd85a8cf37948780c31`

검증 결과:

- 247,435행
- sample submission ID 및 순서 완전 일치
- 중복 ID 0
- 결측 예측 0
- 유효 라벨만 포함
- 원본 193과 바이트 단위 동일
- 저장된 test probability argmax와 전 행 일치

클래스별 OOF recall:

- GALAXY: 0.9602760411
- QSO: 0.9772073449
- STAR: 0.9744934964

이 파일은 public leaderboard 점수를 선택 기준으로 사용하지 않았습니다. paired fold, 취약 subset, nested blend, 외부 SDSS, 추가 stacker ablation까지 통과한 후보 중 최종 유지된 private 일반화 트랙입니다.

검증 보고서:

- `artifacts/final_private_submission_20260630/final_private_submission_report.json`
