<template>
  <div>
    <canvas
      ref='canvas'
      id='smiley-canvas'
      class='smiley-canvas'
      :width='width'
      :height='height'
      v-on:mousedown='startDraw'
      v-on:mouseup='endDraw'
      v-on:mousemove='draw'>
      Sorry! Your browser doesn't seem to support the canvas element. Please try to view
      this on a newer browser!
    </canvas>
  </div>
</template>

<script>
export default {
  name: 'SmileyCanvas',
  props: {
    width: {
      type: Number,
      default: 200
    },
    height: {
      type: Number,
      default: 200
    },
    strokeWidth: {
      type: Number,
      default: 4
    }
  },
  computed: {
    ctx () {
      return this.$refs.canvas.getContext('2d')
    },
    base64Png () {
      return this.$refs.canvas.toDataUrl('image/png')
    }
  },
  data () {
    return {
      canDraw: false
    }
  },
  methods: {
    startDraw () {
      this.canDraw = true
      this.ctx.beginPath()
    },
    endDraw () {
      this.canDraw = false
    },
    draw (e) {
      if (this.canDraw) {
        this.ctx.lineTo(e.layerX, e.layerY)
        this.ctx.stroke()
      }
    },
    clear () {
    }
  },
  mounted () {
    this.ctx.lineWidth = this.strokeWidth
  }
}
</script>

<style scoped>
.smiley-canvas {
  border: 2px solid #111;
  cursor: crosshair;
}
</style>
