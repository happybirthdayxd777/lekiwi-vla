# Task List: Hybrid VLA Robot (Two Modes)

- [x] **Phase 1: Research & Requirements**
    - [x] Analyze provided reference (Manmaru Note).
    - [x] Research OpenVLA/Pi0 hosting options.
    - [x] Analyze Remote Brain architecture.
    - [x] Finalize Open Source vs Gemini trade-offs.
    - [x] Finalize Two-Mode Architecture. <!-- id: 0 -->
    - [/] Finalize Switching Mechanism (Simpler). <!-- id: 0.1 -->
    - [/] Align Mode 2 with xLeRobot (Function Calling). <!-- id: 0.2 -->

- [ ] **Phase 2: Documentation**
    - [ ] Update `requirements.md` (Gemini Tools). <!-- id: 1 -->
    - [ ] Update `implementation_plan.md` (Keyboard/API Switch). <!-- id: 2 -->
    - [ ] Update `operations_guide.md`. <!-- id: 3 -->

- [ ] **Phase 3: Implementation - Server (Akamai)**
    - [ ] Implement `server/vla_server.py` (Hosting Pi0 + LLaVA).
    - [ ] Implement `server/Dockerfile`.

- [ ] **Phase 4: Implementation - Client (Jetson)**
    - [ ] Implement `client/hybrid_client.py` (Mode Switching Logic).
    - [ ] Implement `client/audio_processor.py` (Whisper).
    - [ ] Implement `client/gemini_agent.py` (Persona Mode).
    - [ ] Implement `client/kamai_agent.py` (Worker Mode).

- [ ] **Phase 5: Verification**
    - [ ] Benchmarking Mode 1 vs Mode 2.
