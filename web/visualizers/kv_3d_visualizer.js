// kv_3d_visualizer.js
// 🟡 3D STRUCTURAL KV CACHE TENSOR VISUALIZER (Three.js + WebGL)
// Scientific Clarification: The 3D geometry visualizes the structural tensor dimensions of the cached representation
// (Layers L × KV Channels/Heads × Sequence Length T). The VRAM calculation additionally accounts for representation width,
// batch size, and bytes per element. It is not a literal physical map of GPU memory addresses.

export class KV3DVisualizer {
  /**
   * @param {HTMLElement} container - DOM container element
   */
  constructor(container) {
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    this.gridGroup = null;
    this.tensorGroup = null;
    this.labelsGroup = null;

    this.currentState = null;

    this._initThree();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  _initThree() {
    const THREE = window.THREE;
    if (!THREE) {
      console.error('Three.js is not loaded');
      return;
    }

    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 450;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(36, 28, 42);

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
      this.controls.maxDistance = 150;
      this.controls.minDistance = 8;
      this.controls.target.set(0, 8, 0);
    }

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(20, 50, 30);
    this.scene.add(dirLight);

    // 6. Base Ground Plane
    const gridHelper = new THREE.GridHelper(50, 25, 0xcccccc, 0xeeeeee);
    gridHelper.position.y = 0;
    this.scene.add(gridHelper);

    // 7. Groups
    this.tensorGroup = new THREE.Group();
    this.scene.add(this.tensorGroup);

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
    this.camera.position.set(36, 28, 42);
    this.controls.target.set(0, 8, 0);
    this.controls.update();
  }

  /**
   * Update 3D visualization to match current VRAM sandbox state.
   * @param {Object} state - Output from calculateKVCacheMemory()
   */
  updateState(state) {
    this.currentState = state;
    this.renderTensor();
  }

  renderTensor() {
    const THREE = window.THREE;
    if (!THREE || !this.currentState) return;

    const { model, seqLen, architectureType } = this.currentState;

    // Clear previous geometry
    while (this.tensorGroup.children.length > 0) {
      const obj = this.tensorGroup.children.pop();
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    }

    const L = model.layers;
    const T = seqLen;

    // Layer vertical distribution parameters
    // We visualize each layer as a discrete horizontal shelf
    // For large layer counts (e.g. 80, 96), we scale vertical spacing
    const totalHeight = 18.0;
    const ySpacing = totalHeight / L;

    // Sequence length scaling (normalized to max ~131,072)
    const maxT = 131072;
    const lengthProgress = Math.min(1.0, Math.max(0.04, Math.log10(T) / Math.log10(maxT)));
    const maxBoxLength = 22.0;
    const currentBoxLength = maxBoxLength * lengthProgress;

    if (architectureType === 'conventional') {
      // Conventional MHA / GQA: (Layers L, KV Heads n_KV, Sequence T)
      const nKV = model.kvHeads;
      // Lateral width for KV heads
      const headWidth = Math.min(1.8, 16.0 / nKV);
      const headGap = headWidth * 0.25;
      const totalWidth = nKV * (headWidth + headGap);
      const xStart = -totalWidth / 2 + headWidth / 2;

      // Layer shelves & head blocks
      for (let l = 0; l < L; l++) {
        const y = (l + 0.5) * ySpacing;

        // Subtle layer plane outline
        const planeGeom = new THREE.PlaneGeometry(totalWidth + 1.5, maxBoxLength + 1.0);
        const planeMat = new THREE.MeshBasicMaterial({
          color: 0xf5f5f5,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.25
        });
        const plane = new THREE.Mesh(planeGeom, planeMat);
        plane.rotation.x = Math.PI / 2;
        plane.position.set(0, y - ySpacing * 0.4, 0);
        this.tensorGroup.add(plane);

        // Blocks for each KV head
        for (let h = 0; h < nKV; h++) {
          const x = xStart + h * (headWidth + headGap);

          // Allocated active KV buffer (solid dark block)
          const activeGeom = new THREE.BoxGeometry(headWidth, ySpacing * 0.7, currentBoxLength);
          const activeMat = new THREE.MeshLambertMaterial({
            color: 0x111111,
            transparent: true,
            opacity: 0.85
          });
          const activeMesh = new THREE.Mesh(activeGeom, activeMat);
          activeMesh.position.set(x, y, -maxBoxLength / 2 + currentBoxLength / 2);
          this.tensorGroup.add(activeMesh);

          // Unallocated context capacity wireframe (ghost bounding box)
          const wireGeom = new THREE.BoxGeometry(headWidth, ySpacing * 0.7, maxBoxLength);
          const wireEdges = new THREE.EdgesGeometry(wireGeom);
          const wireMat = new THREE.LineBasicMaterial({
            color: 0xdddddd,
            transparent: true,
            opacity: 0.35
          });
          const wireMesh = new THREE.LineSegments(wireEdges, wireMat);
          wireMesh.position.set(x, y, 0);
          this.tensorGroup.add(wireMesh);
        }
      }
    } else if (architectureType === 'mla') {
      // DeepSeek-V2 MLA: (Layers L, Compressed Latent Channel [512D + 64D], Sequence T)
      // Visualized as a single unified compressed latent slab per layer, distinctly styled!
      const slabWidth = 14.0;

      for (let l = 0; l < L; l++) {
        const y = (l + 0.5) * ySpacing;

        // Active MLA latent block (split visually into 512D latent + 64D RoPE key)
        const latentWidth = slabWidth * (512 / 576);
        const ropeWidth = slabWidth * (64 / 576);

        // 1. 512D KV Latent block (Dark charcoal)
        const latentGeom = new THREE.BoxGeometry(latentWidth, ySpacing * 0.7, currentBoxLength);
        const latentMat = new THREE.MeshLambertMaterial({
          color: 0x222222,
          transparent: true,
          opacity: 0.9
        });
        const latentMesh = new THREE.Mesh(latentGeom, latentMat);
        latentMesh.position.set(-slabWidth / 2 + latentWidth / 2, y, -maxBoxLength / 2 + currentBoxLength / 2);
        this.tensorGroup.add(latentMesh);

        // 2. 64D Decoupled RoPE Key block (Slightly lighter with crisp boundary)
        const ropeGeom = new THREE.BoxGeometry(ropeWidth, ySpacing * 0.7, currentBoxLength);
        const ropeMat = new THREE.MeshLambertMaterial({
          color: 0x555555,
          transparent: true,
          opacity: 0.95
        });
        const ropeMesh = new THREE.Mesh(ropeGeom, ropeMat);
        ropeMesh.position.set(slabWidth / 2 - ropeWidth / 2, y, -maxBoxLength / 2 + currentBoxLength / 2);
        this.tensorGroup.add(ropeMesh);

        // Wireframe ghost bounding box for full context
        const wireGeom = new THREE.BoxGeometry(slabWidth, ySpacing * 0.7, maxBoxLength);
        const wireEdges = new THREE.EdgesGeometry(wireGeom);
        const wireMat = new THREE.LineBasicMaterial({
          color: 0xcccccc,
          transparent: true,
          opacity: 0.35
        });
        const wireMesh = new THREE.LineSegments(wireEdges, wireMat);
        wireMesh.position.set(0, y, 0);
        this.tensorGroup.add(wireMesh);
      }
    }

    // 3D Axis indicators
    this._createAxisIndicators(maxBoxLength, totalHeight);
  }

  _createAxisIndicators(maxZ, maxY) {
    const THREE = window.THREE;
    const origin = new THREE.Vector3(-12, 0, maxZ / 2 + 3);

    // Z-Axis: Sequence Length T (forward-backward)
    const zDir = new THREE.Vector3(0, 0, -1);
    const zArrow = new THREE.ArrowHelper(zDir, origin, maxZ + 2, 0x000000, 1.2, 0.6);
    this.tensorGroup.add(zArrow);

    // Y-Axis: Layers L (vertical)
    const yDir = new THREE.Vector3(0, 1, 0);
    const yArrow = new THREE.ArrowHelper(yDir, origin, maxY + 2, 0x000000, 1.2, 0.6);
    this.tensorGroup.add(yArrow);

    // X-Axis: KV Channels / Heads (lateral)
    const xDir = new THREE.Vector3(1, 0, 0);
    const xArrow = new THREE.ArrowHelper(xDir, origin, 18, 0x000000, 1.2, 0.6);
    this.tensorGroup.add(xArrow);
  }

  _animate() {
    requestAnimationFrame(this._animate);
    if (this.controls) this.controls.update();
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
