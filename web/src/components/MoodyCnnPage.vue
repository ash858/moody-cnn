<template>
  <div>
    Hello hi
    <SmileyCanvas :strokeWidth='4'></SmileyCanvas>
    <button>Clear</button>
  </div>
</template>

<script>
import SmileyCanvas from './SmileyCanvas.vue'

export default {
  components: {
    SmileyCanvas
  }
}
</script>
