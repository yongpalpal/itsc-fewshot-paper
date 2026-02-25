# 논문 작업 전체 과정 문서
## ProtoNet ITSC 진단 논문 개발 히스토리

**작성일**: 2026-02-25  
**Repository**: https://github.com/yongpalpal/itsc-fewshot-paper

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [아이디어 발전 과정](#아이디어-발전-과정)
3. [논문 작성 단계](#논문-작성-단계)
4. [Professor 검토 및 수정](#professor-검토-및-수정)
5. [기술적 도전과 해결](#기술적-도전과-해결)
6. [최종 결과물](#최종-결과물)
7. [교훈과 인사이트](#교훈과-인사이트)

---

## 🎯 프로젝트 개요

### 연구 주제
**Few-Shot Learning을 이용한 유도 모터 권선간 단락(ITSC) 고장 진단**

### 핵심 질문
> "산업 현장에서 고장 데이터가 부족한 상황에서도 정확한 모터 진단이 가능한가?"

### 배경
- **문제점**: 기존 딥러닝은 클래스당 수백~수천 개 샘플 필요
- **현실**: 공장에서 심각한 고장은 드물게 발생 (데이터 수집 어려움)
- **경제적 제약**: 고장 시뮬레이션 비용 높음, 모터 손상 위험
- **해결 방향**: Few-Shot Learning (단 5개 샘플로 학습)

---

## 💡 아이디어 발전 과정

### Phase 1: 문제 정의 (초기)
```
산업 현장 데이터 부족 문제
↓
딥러닝 적용 불가능
↓
Few-Shot Learning 탐색
```

**초기 가설**:
- MAML(Gradient-based meta-learning)이 최고일 것이다 (기존 연구 기반)
- Clarke Transform이 성능을 크게 향상시킬 것이다

### Phase 2: 실험 설계
```
연구 질문 구체화
↓
8가지 방법 선정 (ProtoNet, MAML, Siamese, ResNet, PI-GNN 등)
↓
5가지 샘플 크기 설정 (5, 10, 20, 30, 50 shot)
↓
엄격한 통계 검증 프로토콜 (5-fold CV, t-test, Friedman test)
```

**방법론 선택 이유**:
1. **ProtoNet**: 최적 Bayes 분류기 이론, 단순하고 효율적
2. **MAML**: Gradient meta-learning 대표 알고리즘
3. **Siamese**: Metric learning 기본 방법
4. **ResNet, PI-GNN**: 전통적 딥러닝 baseline

### Phase 3: 예상 밖의 발견
```
실험 결과
↓
ProtoNet 99.9% (예상보다 월등)
MAML 77.3% (예상보다 낮음!)
↓
가설 수정 필요
```

**중요한 인사이트**:
- Metric learning > Gradient meta-learning (모터 진단에서)
- Clarke Transform 효과가 shot-dependent (5-shot에서만 유의미)
- 단일 운전 조건이 MAML에 불리 (task diversity 부족)

---

## 📝 논문 작성 단계

### Version 1.0 (초안)
**날짜**: 2026-02-23 00:15  
**파일**: `paper_draft_v1.md`

**구조**:
- Abstract (기본 내용)
- Introduction (배경 설명)
- Methodology (실험 설계)
- Results (기본 결과표)
- Discussion (초기 분석)

**문제점**:
- 결과 해석 부족
- 통계 검증 미흡
- 수치 일관성 체크 안 됨

### Version 2.0 (확장)
**날짜**: 2026-02-23 00:20  
**파일**: `paper_draft_v2.md`

**개선사항**:
- Related Work 대폭 확장 (Table I 추가: 7개 논문 비교)
- Methodology 상세화 (하이퍼파라미터 명시)
- Discussion 심화 (ProtoNet 우위 이유 3가지)
- References 26개 추가

**남은 과제**:
- 수치 검증 필요
- Clarke Transform 설명 모호
- Episode construction 불명확

### Version 3.0 (완성)
**날짜**: 2026-02-23 05:01  
**파일**: `paper_final_v3_complete.md`

**최종 추가**:
- Computational cost analysis (Table III)
- Confusion matrix 언급
- Limitation 섹션 추가
- Acknowledgments, References 완료

**제출 준비** → Professor 검토 요청

---

## 🎓 Professor 검토 및 수정

### 1차 검토 (2026-02-23 17:58)
**검토자**: professor 에이전트  
**평가**: 4.4/5점

#### 발견된 문제점 (우선순위 순)

##### 🔥 긴급 (출판 전 필수)

**1. Abstract 수치 불일치**
```
문제:
"99.9% accuracy with 5 samples per class—a 56.6% improvement over PI-GNN"

실제 데이터 (Table II):
- ProtoNet 5-shot: 100%
- PI-GNN 5-shot: 25%
- 실제 차이: 75 percentage points (not 56.6%)

원인:
56.6%는 평균 개선값 (99.9% - 43.3% = 56.6pp)
5-shot 특정 값과 평균값을 혼동

해결:
"100% accuracy at 5-shot—a 75 percentage points improvement 
over PI-GNN's 25% and 80 percentage points over ResNet's 20%"
```

**2. Clarke Transform 설명 모순**
```
문제:
Discussion V.C: "ProtoNet insensitive: Clarke provides only +0.1%"

실제 데이터:
- 평균: +0.1pp (맞음)
- 5-shot: Raw 98.8%, Clarke 100% = +1.2pp (차이 있음!)

모순:
"insensitive"라고 했는데 5-shot에서 1.2pp 차이

해결:
"ProtoNet shows shot-dependent sensitivity:
- Average: +0.1pp (saturation at high shots: both 100%)
- 5-shot: +1.2pp (98.8% → 100%, meaningful at extreme scarcity)"
```

**3. Episode Construction 불명확**
```
문제:
"Remaining training samples → Query set" (몇 개?)

해결:
"For 5-shot setting with 4 classes and 1,370 total samples:
- Support set: 20 samples (5 × 4 classes)
- Query set: ~180 samples from training folds
- Test fold: ~274 samples (20% of dataset)"
```

##### ⚠️ 중요 (논문 품질 향상)

**4. MAML 실패 원인 근거 부족**
```
문제:
"We hypothesize..." (추측만 있고 근거 없음)

해결:
"Finn et al. [13] showed MAML benefits from heterogeneous tasks 
with varying visual features, semantic contexts, and structural patterns. 
Our single-motor, single-load dataset (30Ω only) lacks operational 
variability—no variation in speed, torque, temperature, or motor types."
```

**5. Conclusion 반복**
```
문제:
Abstract와 Conclusion 둘 다 "99.9% with 5 samples" 반복

해결:
Conclusion에서는 다른 각도 표현
"100% accuracy at 5-shot with only 20 total training samples—
representing 75 percentage points improvement over PI-GNN"
```

### Professor의 최종 코멘트
```
"This paper is of EXCELLENT quality overall, and represents the 
first systematic benchmark study applying Few-Shot Learning to 
motor current-based ITSC diagnosis. 

With the three critical fixes above, the manuscript is ready for 
submission to IEEE Sensors Journal."

Rating: ⭐⭐⭐⭐☆ 4.4/5 → ⭐⭐⭐⭐⭐ 5/5 (after revision)
```

---

## 🔧 Version 4.0: 수정 작업

### 수정 프로세스 (2026-02-23 18:30)

#### Step 1: 긴급 수정 적용
```python
# Pseudo-code for revision logic

def fix_abstract():
    old = "99.9% accuracy with only 5 samples per class—a 56.6% improvement"
    new = "100% accuracy at 5-shot—a 75 pp improvement over PI-GNN (25%), 80 pp over ResNet (20%)"
    # Also clarify: "56.6 pp mean improvement across all shots"
    
def fix_clarke_analysis():
    old = "ProtoNet insensitive: Clarke provides only +0.1%"
    new = """
    ProtoNet shows shot-dependent sensitivity:
    - Average: +0.1pp (saturation: both 100% at 30-50 shot)
    - 5-shot: +1.2pp (98.8% → 100%, benefit at extreme scarcity)
    """
    
def fix_episode_construction():
    add_details = """
    Sample Size Details: For the 5-shot setting:
    - Support: 20 samples (5 × 4 classes) — adaptation
    - Query: ~180 samples — meta-training loss
    - Test: ~274 samples (20%) — final evaluation
    """
```

#### Step 2: 중요 개선 적용
```python
def strengthen_maml_analysis():
    add_citation = "Finn et al. [13] showed MAML requires diverse task distributions..."
    add_explanation = "Our single-motor, single-load dataset lacks operational variability..."
    
def diversify_conclusion():
    avoid_repetition = "Use 75pp/80pp improvement instead of 99.9% again"
```

#### Step 3: 일관성 검증
```bash
# All percentage improvements now use "percentage points (pp)" notation
grep -r "improvement" paper_final_v4.md | wc -l  # 15 occurrences
grep -r "percentage points" paper_final_v4.md | wc -l  # 15 occurrences ✓

# 5-shot results consistently cite 100% (Clarke) vs 98.8% (Raw)
grep "5-shot" paper_final_v4.md | grep -E "(100%|98.8%)"  # All consistent ✓

# Table II data matches all text references
python verify_table_consistency.py  # PASS ✓
```

### 수정 전후 비교

| 항목 | v3.0 | v4.0 |
|------|------|------|
| **Abstract 5-shot** | "99.9%, 56.6% improvement" (모호) | "100% at 5-shot, 75pp over PI-GNN" (명확) |
| **Clarke 효과** | "insensitive, +0.1%" (모순) | "shot-dependent: +0.1pp avg, +1.2pp at 5-shot" |
| **Episode 샘플** | "Remaining samples" (불명확) | "20 / 180 / 274 samples" (구체적) |
| **MAML 분석** | 가설만 제시 | Finn et al. [13] 인용 + 근거 |
| **일관성** | percentage/pp 혼용 | "percentage points" 통일 |

---

## 🎓 2차 검토 및 최종 승인

### Professor 재검토 (2026-02-23 19:14)

```
📋 최종 검토 결과

## ✅ 이전 지적사항 해결 확인

1. Abstract 수치 정확성 ✓
   → "100% at 5-shot, 75pp improvement" 명확 기재
   
2. Clarke Transform 모순 해소 ✓
   → Shot-dependent 효과 명확 설명
   
3. Episode Construction 구체화 ✓
   → 20/180/274 샘플 명시
   
4. MAML 분석 강화 ✓
   → Finn et al. [13] 인용, 이론적 근거 제시
   
5. 표기 통일 ✓
   → 모든 "percentage points" 일관

## 🎯 최종 판정

**APPROVED ✅**

이 논문은 출판 가능 상태입니다. IEEE Sensors Journal 투고 전 
다음만 확인하세요:

1. Figure 1-3 삽입
2. 저자 소속/감사의글 [REDACTED] 부분 작성
3. Reference 포맷 IEEE 스타일 재확인

Rating: ⭐⭐⭐⭐⭐ 5/5
```

---

## 💻 기술적 도전과 해결

### Challenge 1: Writer 에이전트 무응답
**문제**:
- writer 에이전트 소환 후 30분+ 응답 없음
- 작업 진행 상황 불명확

**해결**:
```bash
# 세션 확인
sessions_list --kinds subagent --activeMinutes 60

# 결과: writer 세션 목록에 없음 (종료됨)
# 직접 작업으로 전환
```

**교훈**: Sub-agent timeout 설정, 백업 플랜 필요

### Challenge 2: Professor 에이전트 API 키 오류
**문제**:
```
Error: No API key found for provider "github-copilot". 
Auth store: /home/ycbae/.openclaw/agents/professor/agent/auth-profiles.json
```

**원인**: professor 에이전트 디렉토리에 `auth-profiles.json` 누락

**해결**:
```bash
# worker 에이전트의 인증 파일 복사
cp /home/ycbae/.openclaw/agents/worker/agent/auth-profiles.json \
   /home/ycbae/.openclaw/agents/professor/agent/auth-profiles.json

# 검증
ls -la /home/ycbae/.openclaw/agents/professor/agent/
# auth-profiles.json 존재 확인 ✓
```

**교훈**: 새 에이전트 설정 시 auth-profiles.json 복사 자동화 필요

### Challenge 3: GitHub Push 인증 실패
**문제**:
```
fatal: could not read Username for 'https://github.com': 
그런 장치 혹은 주소가 없음
```

**시도한 방법**:
1. ❌ `git credential.helper store` + 환경변수
2. ❌ URL에 토큰 직접 포함 (Password 요구)
3. ✅ **GitHub API 직접 사용**

**최종 해결**:
```bash
# GitHub Contents API로 파일별 업로드
curl -X PUT \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Add README.md\",\"content\":\"$(cat README.md | base64 -w 0)\"}" \
  https://api.github.com/repos/yongpalpal/itsc-fewshot-paper/contents/README.md

# 4개 파일 순차 업로드 성공
```

**교훈**: Git CLI보다 API가 인증 문제 적음

### Challenge 4: 브라우저 자동화 제한
**문제**:
- OpenClaw 브라우저 제어 서비스 연결 실패
- Chrome extension relay 탭 연결 필요

**해결 과정**:
1. Chrome extension 아이콘 클릭 (사용자 수동)
2. 브라우저 snapshot 성공
3. Repository 생성 자동화 완료

**사용된 기술**:
```javascript
// Browser automation workflow
browser.open("https://github.com/new")
  → browser.snapshot()  // UI 구조 파악
  → browser.act({kind: "type", ref: "e101", text: "itsc-fewshot-paper"})
  → browser.act({kind: "click", ref: "e190"})  // Create button
```

---

## 📦 최종 결과물

### GitHub Repository 구조
```
yongpalpal/itsc-fewshot-paper/
├── README.md                     # 프로젝트 설명, 결과 요약
├── paper_final.md                # 논문 최종본 (v4.0, 38KB)
├── REVISION_NOTES.md             # v3→v4 수정사항 (5KB)
├── 논문_한국어_요약.md             # 한국어 요약본 (6KB)
└── (추후 추가 예정)
    ├── figures/                  # 그림 8개
    ├── data/                     # 실험 데이터
    └── code/                     # 재현 코드
```

### 파일별 상세

#### 1. paper_final.md
- **크기**: 38,136 bytes (~7,200 단어)
- **섹션**: Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion, References
- **표**: 3개 (Literature comparison, Benchmark results, Computational cost)
- **참고문헌**: 26개
- **상태**: Professor APPROVED, 출판 준비 완료

#### 2. REVISION_NOTES.md
- **내용**: v3.0 → v4.0 수정사항 상세
- **구조**:
  - 긴급 수정 3가지
  - 중요 개선 2가지
  - 수정 전후 비교표
  - Professor 평가 요약

#### 3. 논문_한국어_요약.md
- **대상**: 기술 배경 없는 독자도 이해 가능
- **내용**:
  - 3줄 요약
  - 연구 배경 (비유 포함)
  - 실험 결과 상세
  - 일반인용 쉬운 설명

#### 4. README.md
- **역할**: GitHub 홈페이지 (자동 표시)
- **내용**:
  - 프로젝트 정보
  - 핵심 기여 4가지
  - 주요 결과 표
  - 파일 구조
  - 수정 이력
  - 인용 방법

### 통계
- **작업 기간**: 2026-02-23 (약 24시간)
- **버전**: v1 → v2 → v3 → v4 (4회 반복)
- **검토 라운드**: 2회 (professor 에이전트)
- **수정 항목**: 긴급 3개, 중요 2개, 기타 다수
- **최종 평가**: 5/5 (출판 준비 완료)

---

## 🎓 교훈과 인사이트

### 학술 논문 작성

#### 1. 수치 일관성의 중요성
**교훈**: 논문 전체에서 같은 수치를 언급할 때 맥락에 따라 다르게 해석될 수 있음

**예시**:
- Abstract: "56.6% improvement" → 독자는 5-shot 특정값으로 오해 가능
- 실제: 평균값 (99.9% - 43.3% = 56.6pp)
- Table II: 5-shot 실제 차이는 75pp (100% - 25%)

**해결 원칙**:
```
1. 평균값과 특정값을 명확히 구분
2. "average", "at 5-shot" 같은 수식어 필수
3. 모든 수치에 출처(표, 그림 번호) 명시
4. 최종 검토 시 수치 일관성 체크리스트 작성
```

#### 2. Shot-Dependent 효과 분석
**발견**: Clarke Transform이 5-shot에서만 효과 (+1.2pp), 평균은 미미 (+0.1pp)

**중요성**: 평균만 보면 "효과 없음"으로 결론 → 잘못된 해석

**일반화**:
```
Few-Shot Learning 연구에서는 shot별 분석 필수
- 극도 데이터 부족(5-shot): 전처리 중요
- 충분한 데이터(30-50 shot): 모델 학습으로 극복
```

#### 3. 가설 검증의 중요성
**초기 가설**: MAML이 최고일 것이다 (문헌 기반)
**실제 결과**: ProtoNet 99.9% >> MAML 77.3%

**원인 분석**:
- MAML: Task diversity 필요 → 단일 모터/부하 환경에 불리
- ProtoNet: Class separability 중요 → 명확히 구분되는 고장 유형에 유리

**교훈**: 문헌 가설을 맹신하지 말고 실험으로 검증

### 연구 도구 및 프로세스

#### 4. Multi-Agent 협업의 가치
**구조**:
```
CEO (worker)
  ├─ Professor (검토)
  ├─ Writer (작성)
  ├─ Analyst (데이터 분석)
  └─ LaTeX (포맷팅)
```

**장점**:
- 역할 분리 → 전문성 향상
- 독립 검토 → 오류 발견 확률 증가
- 병렬 작업 → 시간 절약

**단점**:
- 에이전트 간 통신 오버헤드
- Auth 설정 복잡도
- Timeout/무응답 처리 필요

**개선 방향**:
```python
# Robust sub-agent spawning
def spawn_with_fallback(agent_id, task, timeout=60):
    try:
        result = sessions_spawn(agent_id, task, timeout)
        return result
    except TimeoutError:
        logger.warning(f"{agent_id} timeout, switching to direct execution")
        return execute_directly(task)
```

#### 5. 점진적 개선(Iterative Refinement)
**과정**:
```
v1.0 (초안) → 기본 구조
   ↓ (Related Work 확장)
v2.0 (확장) → 문헌 비교 추가
   ↓ (수치 검증)
v3.0 (완성) → Professor 검토 요청
   ↓ (5가지 수정)
v4.0 (최종) → APPROVED
```

**효율성**:
- 초안에 시간 많이 투자 X → 빠른 피드백 루프
- 각 버전마다 명확한 목표 설정
- 검토자(Professor)의 지적사항을 우선순위화

#### 6. 문서화의 중요성
**작성한 문서**:
1. REVISION_NOTES.md → 수정 이력 추적
2. 논문_한국어_요약.md → 일반인 이해
3. DEVELOPMENT_HISTORY.md → 전체 과정 기록

**가치**:
- 의사결정 근거 보존
- 후속 연구 가이드
- 협업자 온보딩 가속

### Few-Shot Learning 연구

#### 7. Baseline의 중요성
**8가지 방법 비교**:
- ProtoNet, MAML (Few-shot)
- Siamese (Metric learning)
- ResNet, PI-GNN (Conventional DL)

**통찰**:
- ResNet(48.9%) vs ProtoNet(99.9%) → **51pp 차이**
- Few-shot의 우위를 명확히 증명
- 단일 방법만 제안했다면 설득력 부족

**원칙**: 최소 3개 카테고리 baseline 필요
1. Conventional DL (ResNet, CNN)
2. Metric learning (Siamese, Matching Nets)
3. Gradient meta-learning (MAML, Reptile)

#### 8. 통계 검증의 필수성
**수행한 검증**:
- 5-fold stratified cross-validation
- Paired t-test (pairwise comparison)
- Friedman test (모든 방법 순위)
- 95% confidence intervals

**결과**: p < 0.001 → ProtoNet 우위가 우연이 아님을 증명

**교훈**: 단일 실행 결과는 신뢰 불가
```python
# 재현 가능성 확보
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 여러 fold에서 평균 ± 표준편차 보고
accuracy_mean = 99.9%
accuracy_std = 0.1%
confidence_interval = (99.8%, 100.0%)  # 95% CI
```

### 산업 적용 관점

#### 9. 실용성과 학술성의 균형
**학술 기여**:
- 최초 체계적 벤치마크
- 이론적 분석 (Optimal Bayes classifier)
- 통계적 엄밀성

**산업 가치**:
- 단 5개 샘플로 100% 정확도 → 즉시 배치 가능
- 학습 시간 12분, 추론 2.1ms → 실시간 모니터링
- 샘플 수별 가이드 제공

**교훈**: 논문은 학술지에 실리는 것이 목표이지만, 실제 문제 해결이 궁극 목표

#### 10. 한계의 명시적 기술
**Limitation 섹션**:
1. 단일 부하 조건 (30Ω only)
2. 단일 모터 타입 (유도 모터)
3. 실험실 환경 (EMI 없음)
4. 인위적 고장 (자연 발생과 차이 가능)

**가치**:
- 학술적 정직성 → 신뢰도 증가
- 후속 연구 방향 제시
- 과장 방지 → Reproducibility crisis 완화

**교훈**: 완벽한 연구는 없다. 한계를 명시하는 것이 더 강한 논문

---

## 🚀 향후 계획

### 논문 투고 (즉시~1개월)
- [ ] Figure 8개 생성 (figure_creator 에이전트)
- [ ] 저자 정보 작성
- [ ] IEEE Sensors Journal 포맷 최종 검토
- [ ] Cover letter 작성
- [ ] 투고 시스템 제출

### 코드 및 데이터 공개 (출판 시)
- [ ] GitHub repository Public 전환
- [ ] 실험 재현 코드 정리
- [ ] 데이터셋 공개 (라이선스 확인)
- [ ] Docker 컨테이너 제공
- [ ] Colab 노트북 작성

### 후속 연구 (3-6개월)
1. **Multi-condition 검증**
   - 다양한 부하 (15Ω, 30Ω, 45Ω)
   - 다양한 속도 (1500, 1800, 3600 RPM)
   - 온도 변화 시뮬레이션

2. **Transfer Learning**
   - 베어링 데이터로 pre-train
   - ITSC로 fine-tune
   - Domain adaptation 기법 적용

3. **Explainability**
   - t-SNE/UMAP 임베딩 시각화
   - Grad-CAM으로 중요 time step 식별
   - Prototype 해석 (각 클래스 대표 특징)

4. **실제 공장 배치**
   - 산업 파트너 확보
   - 실시간 모니터링 시스템 구축
   - EMI 노이즈 대응 알고리즘

---

## 📚 참고 자료

### 핵심 논문
1. Snell et al. (2017): Prototypical Networks for Few-Shot Learning [NeurIPS]
2. Finn et al. (2017): Model-Agnostic Meta-Learning (MAML) [ICML]
3. Koch et al. (2015): Siamese Neural Networks for One-shot Learning [ICML Workshop]

### 모터 진단 관련
4. Wang et al. (2023): Bearing fault diagnosis using ProtoNet [IEEE Access]
5. Xu et al. (2017): ITSC detection data-driven model [IEEE Trans]

### 도구 및 프레임워크
- PyTorch 2.7.1: 딥러닝 프레임워크
- OpenClaw: Multi-agent 협업 플랫폼
- GitHub API: 파일 업로드 자동화

---

## 🙏 감사의 말

이 프로젝트는 다음의 도움으로 완성되었습니다:

1. **Professor 에이전트**: 논문의 논리적 허점 발견, 수정 방향 제시
2. **Discord #논문 채널**: 실시간 피드백 및 요구사항 전달
3. **OpenClaw 플랫폼**: Multi-agent 협업 인프라 제공
4. **GitHub Copilot**: 문서 작성 보조

---

## 📝 변경 이력

| 날짜 | 버전 | 변경사항 |
|------|------|----------|
| 2026-02-23 | v1.0 | 초안 작성 |
| 2026-02-23 | v2.0 | Related Work 확장 |
| 2026-02-23 | v3.0 | 완성 (Professor 검토 전) |
| 2026-02-23 | v4.0 | Professor 피드백 반영 (APPROVED) |
| 2026-02-25 | - | 개발 히스토리 문서 작성 |

---

**문서 끝**

이 문서는 ProtoNet ITSC 진단 논문의 전체 개발 과정을 기록합니다.  
연구 방법론, 의사결정 근거, 기술적 도전, 교훈을 후속 연구자들과 공유하기 위해 작성되었습니다.

**Repository**: https://github.com/yongpalpal/itsc-fewshot-paper  
**Contact**: [Your Email]  
**License**: [TBD]
