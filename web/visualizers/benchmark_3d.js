// benchmark_3d.js
// 🟡 3D ATTENTION MICROSCOPE VISUALIZER (Three.js + WebGL)
// Mechanistic 3D visualization driven by deterministic projections of actual 32D computational state snapshots.
// Scientific Clarification: The 3D scene is a deterministic projection of actual D=32 vectors;
// it is not the original attention space and does not claim to preserve all 32D geometric relationships.

export class AttentionMicroscope3D {
  /**
   * @param {HTMLElement} container - DOM element to attach WebGL renderer
   * @param {Function} onSelectToken - Callback when user selects a token: (tokenInfo) => void
   */
  constructor(container, onSelectToken = null) {
    this.container = container;
    this.onSelectToken = onSelectToken;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.raycaster = null;
    this.mouse = null;

    this.snapshots = null;
    this.currentStep = 1;
    this.currentMode = 'kv_cache'; // 'kv_cache' | 'no_cache'
    this.selectedTokenIndex = 1;

    // Three.js visual objects
    this.keyGroup = null;
    this.rayGroup = null;
    this.queryMesh = null;
    this.outputMesh = null;
    this.gridHelper = null;

    this._initThree();
    this._setupInteraction();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  _initThree() {
    const THREE = window.THREE;
    if (!THREE) {
      console.error('Three.js is not loaded on window.THREE');
      return;
    }

    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 450;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(22, 18, 26);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.container.innerHTML = '';
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    const OrbitControls = THREE.OrbitControls || window.OrbitControls;
    if (OrbitControls) {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.maxDistance = 100;
      this.controls.minDistance = 5;
      this.controls.target.set(0, 0, 0);
    }

    // 5. Subtle Lighting (Monochromatic scientific instrument look)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(20, 40, 20);
    this.scene.add(dirLight);

    // 6. Subtle Technical Ground Grid
    this.gridHelper = new THREE.GridHelper(36, 18, 0xcccccc, 0xeeeeee);
    this.gridHelper.position.y = -10;
    this.scene.add(this.gridHelper);

    // 7. Groups
    this.keyGroup = new THREE.Group();
    this.rayGroup = new THREE.Group();
    this.scene.add(this.keyGroup);
    this.scene.add(this.rayGroup);

    // 8. Query Marker (Distinct Octahedron in Charcoal/Black)
    const qGeom = new THREE.OctahedronGeometry(0.7, 0);
    const qMat = new THREE.MeshBasicMaterial({ color: 0x000000, wireframe: false });
    this.queryMesh = new THREE.Mesh(qGeom, qMat);
    this.queryMesh.visible = false;
    this.scene.add(this.queryMesh);

    // Query Wireframe Outer Ring
    const qRingGeom = new THREE.RingGeometry(0.9, 1.05, 24);
    const qRingMat = new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide });
    this.queryRing = new THREE.Mesh(qRingGeom, qRingMat);
    this.queryRing.visible = false;
    this.scene.add(this.queryRing);

    // 9. Output Marker (Projected Attention Output o_t)
    const outGeom = new THREE.DodecahedronGeometry(0.65, 0);
    const outMat = new THREE.MeshBasicMaterial({ color: 0x555555, wireframe: true });
    this.outputMesh = new THREE.Mesh(outGeom, outMat);
    this.outputMesh.visible = false;
    this.scene.add(this.outputMesh);

    // 10. Raycasting
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points = { threshold: 0.5 };
    this.mouse = new THREE.Vector2(-999, -999);

    // Resize observer
    window.addEventListener('resize', () => this.onResize());
  }

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  resetCamera() {
    if (!this.camera || !this.controls) return;
    this.camera.position.set(22, 18, 26);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setSnapshots(snapshots) {
    this.snapshots = snapshots;
    if (!snapshots || snapshots.length === 0) return;
    this.currentStep = Math.min(this.currentStep, snapshots.length);
    this.selectedTokenIndex = Math.min(this.selectedTokenIndex, this.currentStep);
    this.renderStep(this.currentStep);
  }

  setStep(step) {
    if (!this.snapshots || step < 1 || step > this.snapshots.length) return;
    this.currentStep = step;
    if (this.selectedTokenIndex > step) {
      this.selectedTokenIndex = step;
    }
    this.renderStep(step);
  }

  setMode(mode) {
    this.currentMode = mode; // 'kv_cache' or 'no_cache'
    this.renderStep(this.currentStep);
  }

  setSelectedToken(index) {
    if (index < 1 || index > this.currentStep) return;
    this.selectedTokenIndex = index;
    this.renderStep(this.currentStep);
    this._dispatchSelection();
  }

  _setupInteraction() {
    const canvas = this.renderer.domElement;

    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects(this.keyGroup.children);

      if (intersects.length > 0) {
        const hit = intersects[0].object;
        if (hit.userData && hit.userData.tokenIndex) {
          this.setSelectedToken(hit.userData.tokenIndex);
        }
      }
    });

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects(this.keyGroup.children);
      canvas.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
    });
  }

  _dispatchSelection() {
    if (!this.onSelectToken || !this.snapshots) return;
    const snap = this.snapshots[this.currentStep - 1];
    if (!snap) return;

    const idx = this.selectedTokenIndex;
    const tokenIdx0 = idx - 1;

    const info = {
      step: this.currentStep,
      tokenIndex: idx,
      rawDotProduct: snap.rawDotProducts[tokenIdx0],
      scaledLogit: snap.scaledScores[tokenIdx0],
      attentionWeight: snap.attentionWeights[tokenIdx0],
      keyVector32D: snap.keys[tokenIdx0],
      valueVector32D: snap.values[tokenIdx0],
      projectedKey: snap.projectedKeys[tokenIdx0],
      queryVector32D: snap.query,
      projectedQuery: snap.projectedQuery,
      outputVector32D: snap.outputVector,
      projectedOutput: snap.projectedOutput,
      isNewlyComputed: idx === this.currentStep,
      mode: this.currentMode
    };

    this.onSelectToken(info);
  }

  renderStep(t) {
    const THREE = window.THREE;
    if (!THREE || !this.snapshots) return;

    const snap = this.snapshots[t - 1];
    if (!snap) return;

    // 1. Position Query marker (q_t @ P)
    const pq = snap.projectedQuery;
    this.queryMesh.position.set(pq[0], pq[1], pq[2]);
    this.queryMesh.visible = true;

    this.queryRing.position.set(pq[0], pq[1], pq[2]);
    this.queryRing.lookAt(this.camera.position);
    this.queryRing.visible = true;

    // 2. Position Output marker (o_t @ P)
    const po = snap.projectedOutput;
    this.outputMesh.position.set(po[0], po[1], po[2]);
    this.outputMesh.visible = true;

    // 3. Clear existing key nodes & attention rays
    while (this.keyGroup.children.length > 0) {
      const obj = this.keyGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    while (this.rayGroup.children.length > 0) {
      const obj = this.rayGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    // 4. Render Keys K_1 ... K_t
    const keyGeom = new THREE.SphereGeometry(0.38, 16, 16);
    const selectedGeom = new THREE.SphereGeometry(0.52, 16, 16);

    for (let i = 1; i <= t; i++) {
      const pk = snap.projectedKeys[i - 1];
      const isCurrentToken = i === t;
      const isSelected = i === this.selectedTokenIndex;
      const weight = snap.attentionWeights[i - 1];

      // Material based on Cache Mode and Selection
      let nodeColor = 0x222222;
      let nodeOpacity = 0.85;

      if (this.currentMode === 'no_cache') {
        // No Cache mode: ALL historical projections were recomputed!
        nodeColor = isCurrentToken ? 0x000000 : 0x333333;
      } else {
        // KV Cache mode: Previous tokens 1..t-1 are REUSED (solid stable grey/black),
        // only token t is newly projected.
        nodeColor = isCurrentToken ? 0x000000 : 0x555555;
      }

      if (isSelected) {
        nodeColor = 0x000000;
      }

      const mat = new THREE.MeshBasicMaterial({
        color: nodeColor,
        transparent: true,
        opacity: isSelected ? 1.0 : nodeOpacity
      });

      const mesh = new THREE.Mesh(isSelected ? selectedGeom : keyGeom, mat);
      mesh.position.set(pk[0], pk[1], pk[2]);
      mesh.userData = { tokenIndex: i };
      this.keyGroup.add(mesh);

      // Outer indicator ring for selected or current token
      if (isSelected || isCurrentToken) {
        const ringG = new THREE.RingGeometry(0.65, 0.78, 20);
        const ringM = new THREE.MeshBasicMaterial({
          color: isSelected ? 0x000000 : 0x666666,
          side: THREE.DoubleSide
        });
        const ringMesh = new THREE.Mesh(ringG, ringM);
        ringMesh.position.set(pk[0], pk[1], pk[2]);
        ringMesh.lookAt(this.camera.position);
        this.keyGroup.add(ringMesh);
      }

      // 5. Attention Rays: q_t -> K_i
      // Ray strength, opacity, and line thickness derived directly from alpha_i^(t)
      const rayPoints = [
        new THREE.Vector3(pq[0], pq[1], pq[2]),
        new THREE.Vector3(pk[0], pk[1], pk[2])
      ];
      const rayGeom = new THREE.BufferGeometry().setFromPoints(rayPoints);

      // Normalize weight relative to uniform (1/t) for clear visual contrast
      const alphaRel = weight * t;
      const rayOpacity = Math.max(0.12, Math.min(0.95, weight * 2.5 + 0.1));

      const rayMat = new THREE.LineBasicMaterial({
        color: isSelected ? 0x000000 : 0x444444,
        transparent: true,
        opacity: rayOpacity,
        linewidth: isSelected ? 3 : (alphaRel > 1.2 ? 2 : 1)
      });

      const ray = new THREE.Line(rayGeom, rayMat);
      this.rayGroup.add(ray);
    }

    // 6. Ray from Query to Output vector (illustrating o_t = sum(alpha_i * v_i))
    const outRayPoints = [
      new THREE.Vector3(pq[0], pq[1], pq[2]),
      new THREE.Vector3(po[0], po[1], po[2])
    ];
    const outRayGeom = new THREE.BufferGeometry().setFromPoints(outRayPoints);
    const outRayMat = new THREE.LineDashedMaterial({
      color: 0x000000,
      dashSize: 0.6,
      gapSize: 0.3,
      transparent: true,
      opacity: 0.5
    });
    const outRay = new THREE.Line(outRayGeom, outRayMat);
    outRay.computeLineDistances();
    this.rayGroup.add(outRay);

    this._dispatchSelection();
  }

  _animate() {
    requestAnimationFrame(this._animate);

    if (this.controls) {
      this.controls.update();
    }

    // Gentle orientation of rings toward camera
    if (this.queryRing && this.camera) {
      this.queryRing.lookAt(this.camera.position);
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  destroy() {
    if (this.renderer && this.renderer.domElement) {
      this.container.removeChild(this.renderer.domElement);
      this.renderer.dispose();
    }
  }
}
