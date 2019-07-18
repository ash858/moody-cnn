<template>
  <div>
    <section class="name-input">
      <b-field label="Feature Name">
        <b-input v-model="name" />
      </b-field>
    </section>
    <SmileyCanvas @save="png => add(png)" />
    <section>
      <p>Images: {{ images.length }}</p>
      <img
        v-for="image in images"
        class="train-preview-img"
        :src="image"
        :key="image"
        alt="training image"
      />
    </section>
  </div>
</template>

<script>
import SmileyCanvas from '../components/SmileyCanvas'
import BField from 'buefy/src/components/field/Field'
import BInput from 'buefy/src/components/input/Input'
export default {
  components: {
    BInput,
    BField,
    SmileyCanvas,
  },
  data() {
    return {
      name: '',
      images: [],
    }
  },
  methods: {
    add(png) {
      this.images.push(png)
    },
  },
}
</script>

<style scoped>
.name-input {
  width: 300px;
  margin: auto;
}
.train-preview-img {
  width: 100px;
  height: 100px;
  border: 2px solid #111111;
  border-radius: 3px;
  margin: 20px;
}
</style>
