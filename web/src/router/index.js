import Vue from 'vue'
import Router from 'vue-router'
import MoodyCnnPage from '@/components/MoodyCnnPage.vue'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'MoodyCnnPage',
      component: MoodyCnnPage
    }
  ]
})
