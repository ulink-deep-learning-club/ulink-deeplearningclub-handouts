import { createApp } from 'vue'
import naive from 'naive-ui'
import App from './App.vue'

import "@/assets/styles.css"

const app = createApp(App)

// Use Naive UI
app.use(naive)

app.mount('#app')