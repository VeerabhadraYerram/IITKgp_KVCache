// kv_3d_visualizer.js
// 🟡 3D STRUCTURAL KV CACHE TENSOR VISUALIZER (Three.js 0.160.0 + WebGL)
// Scientific Clarification: The 3D geometry visualizes the structural tensor dimensions of the cached representation
// (Layers L × KV Channels/Heads × Sequence Length T). The VRAM calculation additionally accounts for representation width,
// batch size, and bytes per element. It is not a literal physical map of GPU memory addresses.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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
    this.tensorGroup = null;

    this.currentState = null;

    this._initThree();
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  _initThree() {
    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 450;

    // 1. Scene (transparent, no box)
    this.scene = new THREE.Scene();
    this.scene.background = null;

    // 2. Camera centered directly on the scene origin (0, 0, 0)
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(24, 16, 26);
    this.camera.lookAt(0, 0, 0);

    // 3. Renderer with transparent alpha channel
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';

    this.container.innerHTML = '';
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxDistance = 120;
    this.controls.minDistance = 6;
    this.controls.target.set(0, 0, 0);
    this.controls.update();

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.55);
    dirLight.position.set(20, 40, 25);
    this.scene.add(dirLight);

    // 6. Base Ground Plane (underneath centered tensor)
    const gridHelper = new THREE.GridHelper(36, 18, 0xcccccc, 0xeeeeee);
    gridHelper.position.y = -9;
    this.scene.add(gridHelper);

    // 7. Groups
    this.tensorGroup = new THREE.Group();
    this.scene.add(this.tensorGroup);

    // Resize Observer for robust container sizing
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => this.onResize());
      this.resizeObserver.observe(this.container);
    }
    window.addEventListener('resize', () => this.onResize());

    // Initial resize to settle layout
    setTimeout(() => this.onResize(), 60);
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
    if (!this.camera) return;
    this.camera.position.set(24, 16, 26);
    this.camera.lookAt(0, 0, 0);
    if (this.controls) {
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    }
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
    if (!this.currentState) return;

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

    // Center vertical range from -8 to +8 (totalHeight = 16)
    const totalHeight = 16.0;
    const ySpacing = totalHeight / L;
    const yOffset = -totalHeight / 2;

    // Sequence length scaling (normalized to max ~131,072)
    const maxT = 131072;
    const lengthProgress = Math.min(1.0, Math.max(0.06, Math.log10(T) / Math.log10(maxT)));
    const maxBoxLength = 18.0;
    const currentBoxLength = maxBoxLength * lengthProgress;

    if (architectureType === 'conventional') {
      // Conventional MHA / GQA: (Layers L, KV Heads n_KV, Sequence T)
      const nKV = model.kvHeads;
      const headWidth = Math.min(1.6, 14.0 / nKV);
      const headGap = headWidth * 0.22;
      const totalWidth = nKV * (headWidth + headGap);
      const xStart = -totalWidth / 2 + headWidth / 2;

      for (let l = 0; l < L; l++) {
        const y = yOffset + (l + 0.5) * ySpacing;

        // Layer shelf subtle outline
        const planeGeom = new THREE.PlaneGeometry(totalWidth + 1.2, maxBoxLength + 0.8);
        const planeMat = new THREE.MeshBasicMaterial({
          color: 0xf5f5f5,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.2
        });
        const plane = new THREE.Mesh(planeGeom, planeMat);
        plane.rotation.x = Math.PI / 2;
        plane.position.set(0, y - ySpacing * 0.4, 0);
        this.tensorGroup.add(plane);

        // Blocks for each KV head
        for (let h = 0; h < nKV; h++) {
          const x = xStart + h * (headWidth + headGap);

          // Allocated active KV buffer (solid dark block)
          const activeGeom = new THREE.BoxGeometry(headWidth, ySpacing * 0.72, currentBoxLength);
          const activeMat = new THREE.MeshLambertMaterial({
            color: 0x111111,
            transparent: true,
            opacity: 0.85
          });
          const activeMesh = new THREE.Mesh(activeGeom, activeMat);
          activeMesh.position.set(x, y, -maxBoxLength / 2 + currentBoxLength / 2);
          this.tensorGroup.add(activeMesh);

          // Unallocated context capacity wireframe (ghost bounding box)
          const wireGeom = new THREE.BoxGeometry(headWidth, ySpacing * 0.72, maxBoxLength);
          const wireEdges = new THREE.EdgesGeometry(wireGeom);
          const wireMat = new THREE.LineBasicMaterial({
            color: 0xcccccc,
            transparent: true,
            opacity: 0.35
          });
          const wireMesh = new THREE.LineSegments(wireEdges, wireMat);
          wireMesh.position.set(x, y, 0);
          this.tensorGroup.add(wireMesh);
        }
      }
    } else if (architectureType === 'mla') {
      // DeepSeek-V2 MLA: Compressed Latent Cache [512D + 64D]
      const slabWidth = 14.0;
      const latentWidth = slabWidth * (512 / 576);
      const ropeWidth = slabWidth * (64 / 576);

      for (let l = 0; l < L; l++) {
        const y = yOffset + (l + 0.5) * ySpacing;

        // 1. 512D KV Latent block (Dark charcoal)
        const latentGeom = new THREE.BoxGeometry(latentWidth, ySpacing * 0.75, currentBoxLength);
        const latentMat = new THREE.MeshLambertMaterial({
          color: 0x1a1a1a,
          transparent: true,
          opacity: 0.9
        });
        const latentMesh = new THREE.Mesh(latentGeom, latentMat);
        latentMesh.position.set(-slabWidth / 2 + latentWidth / 2, y, -maxBoxLength / 2 + currentBoxLength / 2);
        this.tensorGroup.add(latentMesh);

        // 2. 64D Decoupled RoPE Key block (Medium gray)
        const ropeGeom = new THREE.BoxGeometry(ropeWidth, ySpacing * 0.75, currentBoxLength);
        const ropeMat = new THREE.MeshLambertMaterial({
          color: 0x555555,
          transparent: true,
          opacity: 0.95
        });
        const ropeMesh = new THREE.Mesh(ropeGeom, ropeMat);
        ropeMesh.position.set(slabWidth / 2 - ropeWidth / 2, y, -maxBoxLength / 2 + currentBoxLength / 2);
        this.tensorGroup.add(ropeMesh);

        // Wireframe ghost bounding box for full context
        const wireGeom = new THREE.BoxGeometry(slabWidth, ySpacing * 0.75, maxBoxLength);
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
    const origin = new THREE.Vector3(-9, -maxY / 2, maxZ / 2 + 1);

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
    const xArrow = new THREE.ArrowHelper(xDir, origin, 15, 0x000000, 1.2, 0.6);
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
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    if (this.renderer && this.renderer.domElement) {
      this.container.removeChild(this.renderer.domElement);
      this.renderer.dispose();
    }
  }
}
