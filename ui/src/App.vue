<template>
  <div id="app">
    <div class="hero-body">
      <div class="container">
        <h1 class="title">
          Moody
        </h1>
      </div>
      <SmileyCanvas />
    </div>
    <router-view />
  </div>
</template>

<script>
import SmileyCanvas from './components/SmileyCanvas'
export default {
  components: { SmileyCanvas },
}
</script>

<style lang="scss">
#app {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  padding: 30px;

  a {
    font-weight: bold;
    color: #2c3e50;

    &.router-link-exact-active {
      color: #42b983;
    }
  }
}
</style>
