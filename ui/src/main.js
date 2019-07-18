import Vue from 'vue'
// Buefy
import Buefy from 'buefy'
import 'buefy/dist/buefy.css'

import App from './App.vue'
import router from './router'
import store from './store'

import { polyfillRequestAnimationFrame } from './utils/animation'

Vue.config.productionTip = false
Vue.use(Buefy)

polyfillRequestAnimationFrame()

new Vue({
  router,
  store,
  render: h => h(App),
}).$mount('#app')
