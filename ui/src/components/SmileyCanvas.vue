<template>
  <div>
    <canvas
      ref="canvas"
      id="smiley-canvas"
      class="smiley-canvas"
      :width="width"
      :height="height"
      @touchstart="startTouchDraw"
      @mousedown="startMouseDraw"
      @touchend="endDraw"
      @mouseup="endDraw"
      @touchmove="touchDraw"
      @mousemove="mouseDraw"
    >
      Sorry! Your browser doesn't seem to support the canvas element. Please try
      to view this on a newer browser!
    </canvas>
    <div class="level is-mobile" style="width: 300px; margin: auto;">
      <div class="level-item has-text-centered">
        <a @click="" class="level-left button is-success is-outlined">
          Accept
        </a>
      </div>
      <div class="level-item has-text-centered">
        <a @click="clear" class="button is-danger is-outlined">
          Reset
        </a>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SmileyCanvas',
  props: {
    width: {
      type: Number,
      default: 300,
    },
    height: {
      type: Number,
      default: 300,
    },
    strokeWidth: {
      type: Number,
      default: 6,
    },
  },
  computed: {
    canv() {
      return this.$refs.canvas
    },
    ctx() {
      return this.canv.getContext('2d')
    },
    base64Png() {
      return this.canv.toDataUrl('image/png')
    },
  },
  data() {
    return {
      canDraw: false,
      currPos: {
        x: 0,
        y: 0,
      },
      lastPos: {
        x: 0,
        y: 0,
      },
    }
  },
  watch: {
    currPos: {
      handler: function() {
        this.requestAnimation()
      },
      deep: true,
    },
  },
  methods: {
    startTouchDraw(e) {
      let pos = this.getTouchPos(e)
      e.preventDefault()
      this.startDraw(pos)
    },
    startMouseDraw(e) {
      let pos = this.getMousePos(e)
      e.preventDefault()
      this.startDraw(pos)
    },
    startDraw(pos) {
      this.lastPos.x = pos.x
      this.lastPos.y = pos.y
      this.canDraw = true
      this.ctx.beginPath()
    },
    endDraw(e) {
      e.preventDefault()
      this.canDraw = false
    },
    touchDraw(e) {
      const touch = e.touches[0]
      const rect = this.canv.getBoundingClientRect()
      this.currPos.x = touch.clientX - rect.left
      this.currPos.y = touch.clientY - rect.top
    },
    mouseDraw(e) {
      const pos = this.getMousePos(e)
      this.currPos.x = pos.x
      this.currPos.y = pos.y
    },
    renderCanvas() {
      if (this.canDraw) {
        this.ctx.moveTo(this.lastPos.x, this.lastPos.y)
        this.ctx.lineTo(this.currPos.x, this.currPos.y)
        this.ctx.stroke()
        this.lastPos.x = this.currPos.x
        this.lastPos.y = this.currPos.y
      }
    },
    getMousePos(e) {
      const rect = this.canv.getBoundingClientRect()
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      }
    },
    getTouchPos(e) {
      const touch = e.touches[0]
      const rect = this.canv.getBoundingClientRect()
      return {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top,
      }
    },
    clear() {
      this.resetPos()
      this.ctx.fillStyle = 'white'
      this.ctx.clearRect(0, 0, this.canv.height, this.canv.width)
    },
    resetPos() {
      this.currPos.x = 0
      this.currPos.y = 0
      this.lastPos.y = 0
      this.lastPos.y = 0
    },
    requestAnimation() {
      if (this.canDraw) {
        window.requestAnimationFrame(() => {
          this.renderCanvas()
        })
      }
    },
  },
  mounted() {
    this.ctx.lineWidth = this.strokeWidth
  },
}
</script>

<style scoped>
.smiley-canvas {
  border: 2px solid #111;
  border-radius: 4px;
  cursor: crosshair;
  margin-top: 20px;
}
</style>
