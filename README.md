# Prototypical Networks for Few-Shot ITSC Diagnosis

**IEEE Sensors Journal 투고 준비 논문**

## 📄 논문 정보

- **제목**: Prototypical Networks for Few-Shot Inter-Turn Short Circuit Diagnosis: A Systematic Benchmark Study
- **버전**: v4.0 (Professor Review 반영 완료)
- **작성일**: 2026-02-23
- **상태**: 출판 준비 완료 (그림 삽입 대기)

## 🎯 핵심 기여

1. **Few-Shot Learning 벤치마크**: 모터 전류 기반 ITSC 진단을 위한 최초의 체계적 비교 연구
2. **100% 5-shot 정확도**: ProtoNet이 단 5개 샘플/클래스로 완벽한 진단 달성
3. **Metric vs. Gradient Meta-Learning**: ProtoNet(99.9%) >> MAML(77.3%) 22.6%p 차이
4. **Clarke Transform 분석**: Shot-dependent 효과 규명 (5-shot: +1.2pp, 평균: +0.1pp)

## 📊 주요 결과

| Method | Average Accuracy | 5-shot | 10-shot | 50-shot |
|--------|------------------|--------|---------|---------|
| **ProtoNet (Clarke)** | **99.9%** | **100%** | **100%** | **100%** |
| ProtoNet (Raw) | 99.8% | 98.8% | 99.6% | 100% |
| MAML (Clarke) | 77.3% | 35% | 73% | 77% |
| Siamese (Clarke) | 69.1% | 55% | 50% | 83% |
| ResNet-1D | 48.9% | 20% | 60% | 63% |
| PI-GNN | 43.3% | 25% | 30% | 55% |

**통계적 검증**: Paired t-test (p < 0.001), Friedman test (χ²=34.8, p<0.001)

## 📁 파일 구조

```
itsc-fewshot-paper/
├── paper_final.md          # 논문 최종본 (v4.0)
├── REVISION_NOTES.md       # Professor 검토 반영 사항
├── README.md               # 이 파일
└── (figures/)              # 그림 파일 (추후 추가)
```

## 🔍 수정 이력

### v4.0 (2026-02-23) - Professor Review 반영
- ✅ Abstract 수치 정확성: "75pp improvement over PI-GNN" 명확화
- ✅ Clarke Transform 분석: Shot-dependent 효과 설명 개선
- ✅ Episode Construction: 샘플 수 구체화 (20/180/274)
- ✅ MAML 분석: 문헌 인용으로 이론적 근거 강화
- ✅ 일관성 검증: 모든 percentage points 표기 통일

### v3.0 (2026-02-23) - 초안 완성
- 8개 방법 벤치마크 결과
- 5-fold CV + 통계 검증
- Table I (문헌 비교), Table II (결과), References 26개

## 🚀 다음 단계

1. **그림 삽입** (8개 예정):
   - Fig. 1: 벤치마크 비교 바 차트
   - Fig. 2: Clarke Transform ablation
   - Fig. 3: Confusion matrix (ProtoNet, 50-shot)
   - Fig. 4-8: 추가 시각화

2. **최종 교정**:
   - IEEE Sensors Journal 포맷팅
   - 참고문헌 스타일 검증
   - 오타 확인

3. **Supplementary Materials** (선택):
   - 하이퍼파라미터 sensitivity 분석
   - 데이터 분포 Gaussian fitting 결과
   - MAML gradient norm 분석

## 📖 인용

```bibtex
@article{ycbae2026protonet,
  title={Prototypical Networks for Few-Shot Inter-Turn Short Circuit Diagnosis: A Systematic Benchmark Study},
  author={[Authors]},
  journal={IEEE Sensors Journal (Under Review)},
  year={2026}
}
```

## 📧 연락처

- **저자**: [Your Name]
- **소속**: [Your Institution]
- **이메일**: [Your Email]

---

**라이선스**: [추후 결정]  
**데이터셋**: 출판 시 공개 예정
